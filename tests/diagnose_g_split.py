"""Bisection g1(A) vs g2(B): который из двух матмулов g_lora ответственен
за катастрофический разрыв real-vs-fake (g_lora_only real=167.25 vs
fake=12.42). Собираем .rwkvq вручную: только выбранный суффикс (.g1 или .g2)
на каждом слое квантуется в INT4, всё остальное dense (bits=16, без потерь)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize
from rwkv_quant.formats.schema import QuantizedTensor, QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536
BITS = 4


def build_ckpt(target_suffix):
    sd = torch.load(CKPT_PTH, map_location="cpu")
    tensors = {}
    for key, w in sd.items():
        if key.endswith(target_suffix):
            codes, scale = _real_quantize(w.float(), BITS)
            tensors[key] = QuantizedTensor(key=key, group="g_lora", bits=BITS,
                                            shape=tuple(w.shape), codes=codes, scale=scale)
        else:
            tensors[key] = QuantizedTensor(key=key, group="other", bits=16,
                                            shape=tuple(w.shape), dense=w.to(torch.bfloat16))
    return QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                                head_size=HEAD_SIZE, vocab_size=VOCAB,
                                tensors=tensors, config_repr=f"only {target_suffix} @ INT4")


def run(name, target_suffix):
    ckpt = build_ckpt(target_suffix)
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
    print(f"{name:20s} ppl={ppl:14.4f}")


if __name__ == "__main__":
    target = sys.argv[1]  # "g1" or "g2"
    run(f"g_only_{target}", f".{target}")
