"""Численная валидация mx.compile на decode: greedy 64 токена eager vs
compiled + относительная ошибка логитов на каждом шаге."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.presets import REDUCTION
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
import rwkv_quant.backends.metal.quant_model as qm

sd = torch.load(os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"), map_location="cpu")
tensors = {k: quantize_tensor(k, w, REDUCTION) for k, w in sd.items()}
del sd
model = qm.QuantRWKV7(QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                       head_size=64, vocab_size=65536, tensors=tensors, config_repr="r"))
np.random.seed(0)
prompt = np.random.randint(0, 65536, size=(1, 16), dtype=np.int64)

def run(step):
    states = model.init_state(1)
    logits, states = model.forward_stateful(mx.array(prompt), states)  # префилл всегда eager
    mx.eval(logits)
    tok = int(mx.argmax(logits[:, -1]).item()); toks = [tok]; outs = [np.array(logits[:, -1])]
    for _ in range(64):
        logits, states = step(mx.array([[tok]]), states)
        mx.eval(logits)
        outs.append(np.array(logits[:, -1]))
        tok = int(mx.argmax(logits[:, -1]).item()); toks.append(tok)
    return np.stack(outs), toks

ref, toks_ref = run(model.forward_stateful)
cmp_, toks_cmp = run(mx.compile(model.forward_stateful))
rel = np.abs(ref - cmp_).max() / (np.abs(ref).max() + 1e-9)
print(f"max rel: {rel:.3e}; greedy совпадает: {toks_ref == toks_cmp}")
div = next((i for i,(a,b) in enumerate(zip(toks_ref,toks_cmp)) if a!=b), None)
print(f"первое расхождение траектории: {div}")
