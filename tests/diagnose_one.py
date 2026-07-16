"""Один диагностический кейс = один процесс (гарантия освобождения памяти
ОС при выходе, вместо полагания на del/gc внутри долгоживущего процесса)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

CASES = {
    # НЕ равен пресету presets.REDUCTION: LoRA-ветки тут НЕ квантованы
    # (bits=16, dense) -- отсюда ppl 11.52 против 13.15 у настоящего
    # REDUCTION (там w/a/v_lora=4, g_lora=8). Кейс для A/B бэкенда, не
    # для оценки качества пресета.
    "reduction_dense_lora": QuantConfig(proj=8, cmix=8, emb_head=8, small=8),
    "proj":       QuantConfig(proj=4, outlier_fracs={"proj": 0.02}),
    "cmix":       QuantConfig(cmix=4, outlier_fracs={"cmix": 0.02}),
    "emb_head":   QuantConfig(emb_head=4, outlier_fracs={"emb_head": 0.02}),
    "small":      QuantConfig(small=6),
    "lora":       QuantConfig(w_lora=4, a_lora=4, v_lora=4, g_lora=4),
    "proj_cmix":  QuantConfig(proj=4, cmix=4, outlier_fracs={"proj": 0.02, "cmix": 0.02}),
    "proj_cmix_embhead": QuantConfig(proj=4, cmix=4, emb_head=4,
                                      outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02}),
    "lora_g8": QuantConfig(w_lora=4, a_lora=4, v_lora=4, g_lora=8),
    "lora_g4_biasfix": QuantConfig(w_lora=4, a_lora=4, v_lora=4, g_lora=4),
    # Открытый вопрос №1, bisection: какая из четырёх LoRA-веток по отдельности
    # даёт основной вклад в разрыв real vs fake_quant (~20x на комбинации всех
    # четырёх). Остальные три ветки держим на bits=16 (dense, без потерь).
    "w_lora_only": QuantConfig(w_lora=4),
    "a_lora_only": QuantConfig(a_lora=4),
    "v_lora_only": QuantConfig(v_lora=4),
    "g_lora_only": QuantConfig(g_lora=4),
    "compression_g4_biasfix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=4,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    "compression_fixed": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    # Кандидат на замену g_lora=8-воркэраунда: g_lora=4 + SpQR (frac=0.02),
    # раз SpQR полностью гасит межслойную нестабильность в изоляции (159.64
    # -> 12.15). Проверяем СОВОКУПНЫЙ эффект вместе с proj/cmix/emb_head/
    # small на INT4/6 -- а не g_lora в вакууме.
    "compression_g4_spqr": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=4,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02, "g_lora": 0.02},
    ),
    "compression_g4_spqr01": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=4,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02, "g_lora": 0.01},
    ),
    "compression_g8_spqr": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02, "g_lora": 0.02},
    ),
    # Итоговый конфиг из calibrate() end-to-end (открытый вопрос №2,
    # tests/diagnose_calibrate_e2e.py): g_lora=6 выбран ИСКЛЮЧИТЕЛЬНО на
    # основании fake_quant/RWKV7Ref (Δ=+0.09% на INT6 fake). Проверяем, не
    # даёт ли эта fake-оценка ложную уверенность -- по аналогии с g_lora=4,
    # где fake предсказывал +8.7%, а реальный пайплайн дал +1363%.
    "calibrated_e2e": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=6,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        clip_percentiles={"small": 99.9},
    ),
}


def main():
    name = sys.argv[1]
    cfg = CASES[name]

    sd = torch.load(CKPT_PTH, map_location="cpu")
    data = torch.load(CORPUS)[:8].numpy()

    tensors = {key: quantize_tensor(key, w, cfg) for key, w in sd.items()}
    ckpt = QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                                head_size=HEAD_SIZE, vocab_size=VOCAB,
                                tensors=tensors, config_repr=repr(cfg))
    del sd
    model = QuantRWKV7(ckpt)

    total_nll, total_tok = 0.0, 0
    batch_size = 4
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i + batch_size]
            idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
            logits = model(idx); mx.eval(logits)
            logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-12))
            B, T, V = logp.shape
            idxf = target.reshape(-1); logpf = logp.reshape(-1, V)
            nll = -logpf[np.arange(len(idxf)), idxf]
            total_nll += nll.sum(); total_tok += nll.size
    ppl = float(np.exp(total_nll / total_tok))
    print(f"{name:20s} ppl={ppl:14.4f}")


if __name__ == "__main__":
    main()
