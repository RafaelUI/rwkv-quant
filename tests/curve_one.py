"""Один конфиг = один процесс (память гарантированно освобождается ОС при
выходе). Пишет результат JSON-строкой в stdout для сборки скриптом-обвязкой."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.presets import REDUCTION, COMPRESSION
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536
PPL_N_SEQ, DECODE_WARMUP, DECODE_MEASURE = 8, 5, 30

CANONICAL_RTN = QuantConfig(   # тот же битовый бюджет, что COMPRESSION (после фикса g_lora=8), но БЕЗ outlier_fracs
    proj=4, cmix=4, emb_head=4,
    w_lora=4, a_lora=4, v_lora=4, g_lora=8,
    small=6, clip_percentiles={"small": 99.9},
)

CONFIGS = {
    "baseline": QuantConfig(),
    "reduction": REDUCTION,
    "compression": COMPRESSION,
    "canonical_rtn": CANONICAL_RTN,
}


def measure_size_mb(ckpt):
    total = 0
    for qt in ckpt.tensors.values():
        if qt.bits < 16:
            total += qt.codes.numel() * 1 + qt.scale.numel() * 2
            if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
                total += qt.outlier_indices.numel() * 4 + qt.outlier_values.numel() * 2
        else:
            total += qt.dense.numel() * 2
    return total / 1e6


@torch.no_grad()
def compute_ppl(model, data_np, batch_size=4):
    total_nll, total_tok = 0.0, 0
    for i in range(0, data_np.shape[0], batch_size):
        batch = data_np[i:i+batch_size]
        idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
        logits = model(idx); mx.eval(logits)
        logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1)+1e-12))
        B,T,V = logp.shape
        idxf = target.reshape(-1); logpf = logp.reshape(-1,V)
        nll = -logpf[np.arange(len(idxf)), idxf]
        total_nll += nll.sum(); total_tok += nll.size
    return float(np.exp(total_nll/total_tok))


def measure_tokens_per_sec(model, n_warmup=DECODE_WARMUP, n_measure=DECODE_MEASURE):
    states = model.init_state(batch_size=1)
    tok = mx.array([[0]])
    for _ in range(n_warmup):
        logits, states = model.forward_stateful(tok, states); mx.eval(logits)
        tok = mx.argmax(logits[:, -1], axis=-1, keepdims=True); mx.eval(tok)
    t0 = time.time()
    for _ in range(n_measure):
        logits, states = model.forward_stateful(tok, states); mx.eval(logits)
        tok = mx.argmax(logits[:, -1], axis=-1, keepdims=True); mx.eval(tok)
    return n_measure / (time.time() - t0)


def main():
    name = sys.argv[1]
    cfg = CONFIGS[name]

    sd = torch.load(CKPT_PTH, map_location="cpu")
    data = torch.load(CORPUS)[:PPL_N_SEQ].numpy()

    tensors = {key: quantize_tensor(key, w, cfg) for key, w in sd.items()}
    ckpt = QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                                head_size=HEAD_SIZE, vocab_size=VOCAB,
                                tensors=tensors, config_repr=repr(cfg))
    size_mb = measure_size_mb(ckpt)
    del sd

    model = QuantRWKV7(ckpt)
    tps = measure_tokens_per_sec(model)
    ppl = compute_ppl(model, data)

    result = {"name": name, "size_mb": size_mb, "tokens_per_sec": tps, "ppl": ppl}
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
