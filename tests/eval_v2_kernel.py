"""ppl реального .rwkvq через кернельный путь (GwQuantLinear, без
разворачивания в dense) + decode-смок: жадная траектория 48 токенов
против dense-деквант модели + скорость мс/ток."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw, _dequantize_one
from rwkv_quant.formats.schema import QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
ckpt = load_raw("/tmp/champion_v2.rwkvq")
model = QuantRWKV7(ckpt)

data = torch.load(CORPUS)[:8].numpy()
total_nll, total_tok = 0.0, 0
for i in range(0, data.shape[0], 4):
    batch = data[i:i+4]
    idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
    logits = model(idx); mx.eval(logits)
    logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-12))
    V = logp.shape[-1]
    nll = -logp.reshape(-1, V)[np.arange(target.size), target.reshape(-1)]
    total_nll += nll.sum(); total_tok += nll.size
print(f"champion_v2_kernel  ppl={float(np.exp(total_nll/total_tok)):14.4f}")

# --- decode: жадная траектория + скорость ---
ckpt2 = load_raw("/tmp/champion_v2.rwkvq")
dense = {k: QuantizedTensor(key=k, group=qt.group, bits=16, shape=qt.shape,
                             dense=_dequantize_one(qt) if qt.bits < 16 else qt.dense)
         for k, qt in ckpt2.tensors.items()}
ckpt2.tensors = dense
ref_model = QuantRWKV7(ckpt2)

prompt = mx.array(data[0:1, :64].astype(np.int32))
def greedy(m, n=48):
    st = m.init_state(1)
    logits, st = m.forward_stateful(prompt, st, last_only=True)
    toks = []
    for _ in range(n):
        t = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(t); toks.append(int(t[0]))
        logits, st = m.forward_stateful(t[None], st)
    return toks

tk, tr = greedy(model), greedy(ref_model)
match = sum(a == b for a, b in zip(tk, tr))
print(f"greedy match vs dense-dequant: {match}/48")

st = model.init_state(1)
logits, st = model.forward_stateful(prompt, st, last_only=True)
tok = mx.argmax(logits[:, -1], axis=-1)
for _ in range(8):   # прогрев compile
    logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
t0 = time.time(); n = 64
for _ in range(n):
    logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
print(f"decode: {(time.time()-t0)/n*1000:.2f} ms/tok")
