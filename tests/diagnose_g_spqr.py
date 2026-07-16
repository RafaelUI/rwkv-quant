"""Проверка практического фикса: спасает ли SpQR (outlier_fracs) от
экспоненциальной межслойной нестабильности g_lora (конкретно g2), которую
обнаружили в diagnose_g_layer_position.py (11.43 -> 159.64 на 24 слоях
чистого RTN INT4). Квантуем g1+g2 во ВСЕХ слоях через SpQR вместо RTN,
разные outlier_frac."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize, _real_quantize_sparse_outlier
from rwkv_quant.formats.schema import QuantizedTensor, QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536
BITS = 4


def build_ckpt(outlier_frac, quantize_g1_too):
    sd = torch.load(CKPT_PTH, map_location="cpu")
    tensors = {}
    for key, w in sd.items():
        is_g2 = key.endswith(".g2")
        is_g1 = key.endswith(".g1")
        if is_g2 or (is_g1 and quantize_g1_too):
            if outlier_frac is not None:
                codes, scale, oi, ov = _real_quantize_sparse_outlier(w.float(), BITS, outlier_frac)
                tensors[key] = QuantizedTensor(key=key, group="g_lora", bits=BITS,
                                                shape=tuple(w.shape), codes=codes, scale=scale,
                                                outlier_indices=oi, outlier_values=ov)
            else:
                codes, scale = _real_quantize(w.float(), BITS)
                tensors[key] = QuantizedTensor(key=key, group="g_lora", bits=BITS,
                                                shape=tuple(w.shape), codes=codes, scale=scale)
        else:
            tensors[key] = QuantizedTensor(key=key, group="other", bits=16,
                                            shape=tuple(w.shape), dense=w.to(torch.bfloat16))
    return QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                                head_size=HEAD_SIZE, vocab_size=VOCAB,
                                tensors=tensors, config_repr=f"g2(+g1={quantize_g1_too}) SpQR frac={outlier_frac}")


def run(name, outlier_frac, quantize_g1_too=False):
    ckpt = build_ckpt(outlier_frac, quantize_g1_too)
    model = QuantRWKV7(ckpt)
    data = torch.load(CORPUS)[:8].numpy()
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
    print(f"{name:30s} ppl={ppl:14.4f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "spqr02":
        run("g2_spqr_frac0.02", 0.02)
    elif mode == "spqr01":
        run("g2_spqr_frac0.01", 0.01)
    elif mode == "spqr005":
        run("g2_spqr_frac0.005", 0.005)
    elif mode == "spqr02_g1too":
        run("g1+g2_spqr_frac0.02", 0.02, quantize_g1_too=True)
