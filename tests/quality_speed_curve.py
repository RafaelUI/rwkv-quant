"""
Кривая "качество (ppl) vs скорость генерации (tokens/sec)" на реальном
rwkv7-g1h-1.5b, через настоящий backends/metal/quant_model.QuantRWKV7
(не fake_quant-симуляцию) -- и ppl, и tok/s измеряются на том же коде,
что реально исполняется.

Квантование делается in-memory (formats.writer.quantize_tensor напрямую
по state_dict), без записи .rwkvq на диск -- экономим место (11GB free).

Конфиги: baseline (без квантования), REDUCTION, COMPRESSION (presets.py),
и "canonical RTN" на размере COMPRESSION (те же биты, без outlier_fracs) --
контрольная точка, показывающая цену отказа от SpQR при том же сжатии.
"""
import sys, os, time, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import mlx.core as mx

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.presets import REDUCTION, COMPRESSION
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

CANONICAL_RTN = QuantConfig(
    proj=4, cmix=4, emb_head=4,
    w_lora=4, a_lora=4, v_lora=4, g_lora=4, small=6,
    clip_percentiles={"small": 99.9},
)  # тот же битовый бюджет, что COMPRESSION, но БЕЗ outlier_fracs

CONFIGS = [
    ("baseline (bf16)", QuantConfig()),
    ("REDUCTION", REDUCTION),
    ("COMPRESSION", COMPRESSION),
    ("canonical RTN (=COMPRESSION size, no SpQR)", CANONICAL_RTN),
]

PPL_N_SEQ = 8          # подмножество eval_corpus_world для скорости прогона
DECODE_WARMUP = 5
DECODE_MEASURE = 30


def build_checkpoint_in_memory(sd, cfg):
    tensors = {key: quantize_tensor(key, w, cfg) for key, w in sd.items()}
    return QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                                head_size=HEAD_SIZE, vocab_size=VOCAB,
                                tensors=tensors, config_repr=repr(cfg))


def measure_size_mb(ckpt):
    total = 0
    for qt in ckpt.tensors.values():
        if qt.bits < 16:
            total += qt.codes.numel() * 1
            total += qt.scale.numel() * 2
            if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
                total += qt.outlier_indices.numel() * 4
                total += qt.outlier_values.numel() * 2
        else:
            total += qt.dense.numel() * 2
    return total / 1e6


@torch.no_grad()
def compute_ppl(model, data_np, batch_size=4):
    total_nll, total_tok = 0.0, 0
    for i in range(0, data_np.shape[0], batch_size):
        batch = data_np[i:i + batch_size]
        idx = mx.array(batch[:, :-1])
        target = batch[:, 1:]
        logits = model(idx)
        mx.eval(logits)
        logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-12))
        B, T, V = logp.shape
        idx_flat = target.reshape(-1)
        logp_flat = logp.reshape(-1, V)
        nll = -logp_flat[np.arange(len(idx_flat)), idx_flat]
        total_nll += nll.sum()
        total_tok += nll.size
    return float(np.exp(total_nll / total_tok))


def measure_tokens_per_sec(model, n_warmup=DECODE_WARMUP, n_measure=DECODE_MEASURE):
    states = model.init_state(batch_size=1)
    tok = mx.array([[0]])
    for _ in range(n_warmup):
        logits, states = model.forward_stateful(tok, states)
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
        mx.eval(tok)
    t0 = time.time()
    for _ in range(n_measure):
        logits, states = model.forward_stateful(tok, states)
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
        mx.eval(tok)
    dt = time.time() - t0
    return n_measure / dt


def main():
    print("=== загрузка 1.5B checkpoint (world naming) ===")
    t0 = time.time()
    sd = torch.load(CKPT_PTH, map_location="cpu")
    print(f"loaded in {time.time()-t0:.1f}s, {len(sd)} tensors")

    data = torch.load(CORPUS)[:PPL_N_SEQ].numpy()
    print(f"eval corpus subset: {data.shape}")

    results = []
    for name, cfg in CONFIGS:
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        t0 = time.time()
        ckpt = build_checkpoint_in_memory(sd, cfg)
        size_mb = measure_size_mb(ckpt)
        print(f"quantized in {time.time()-t0:.1f}s, size={size_mb:.1f}MB")

        t0 = time.time()
        model = QuantRWKV7(ckpt)
        print(f"model built in {time.time()-t0:.1f}s")

        t0 = time.time()
        tps = measure_tokens_per_sec(model)
        print(f"decode speed: {tps:.2f} tok/s  [{time.time()-t0:.1f}s measured]")

        t0 = time.time()
        ppl = compute_ppl(model, data)
        print(f"ppl: {ppl:.4f}  [{time.time()-t0:.1f}s]")

        results.append({"name": name, "size_mb": size_mb, "tokens_per_sec": tps, "ppl": ppl})

        del model, ckpt
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    print(f"\n\n{'='*70}\nСВОДКА\n{'='*70}")
    baseline_ppl = results[0]["ppl"]
    for r in results:
        delta = 100 * (r["ppl"] - baseline_ppl) / baseline_ppl
        print(f"{r['name']:44s} ppl={r['ppl']:9.4f} (Δ{delta:+7.2f}%)  "
              f"size={r['size_mb']:8.1f}MB  speed={r['tokens_per_sec']:7.2f} tok/s")

    with open("/tmp/quality_speed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nсохранено в /tmp/quality_speed_results.json")


if __name__ == "__main__":
    main()
