"""Гейт фьюза: (1) relmax логитов fused/unfused на префилле и в decode,
(2) greedy 48 токенов идентичен, (3) ppl корпуса через ФЬЮЗНУТЫЙ
forward_stateful == референс кернельного пути (11.7125), (4) A/B-скорость
compiled fused vs unfused."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

model = qm.QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
data = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt"))[:8].numpy()
prompt = mx.array(data[0:1, :64].astype(np.int32))

# --- 1. паритет логитов: префилл T=64 + 8 decode-шагов, raw путь ---
def run_traj(fuse):
    qm.FUSE = fuse
    st = model.init_state(1)
    logits, st = model.forward_stateful(prompt, st, last_only=True)
    outs = [logits]
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(8):
        logits, st = model.forward_stateful(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        outs.append(logits)
    mx.eval(*outs)
    return [np.array(o.astype(mx.float32)) for o in outs]

a, b = run_traj(False), run_traj(True)
rel = max(float(np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-9)) for x, y in zip(a, b))
print(f"parity relmax (prefill+8 decode): {rel:.2e}")

# --- 2. greedy 48 ---
def greedy(fuse, n=48):
    qm.FUSE = fuse
    st = model.init_state(1)
    logits, st = model.forward_stateful(prompt, st, last_only=True)
    toks = []
    for _ in range(n):
        t = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(t); toks.append(int(t[0]))
        logits, st = model.forward_stateful(t[None], st)
    return toks
ga, gb = greedy(False), greedy(True)
print(f"greedy fused vs unfused: {sum(x==y for x,y in zip(ga,gb))}/48")

# --- 3. ppl через фьюзнутый forward_stateful (прямой прогон корпуса) ---
qm.FUSE = True
total_nll, total_tok = 0.0, 0
for i in range(0, data.shape[0], 4):
    batch = data[i:i+4]
    idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
    logits, _ = model.forward_stateful(idx, model.init_state(batch.shape[0]))
    mx.eval(logits)
    logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-12))
    V = logp.shape[-1]
    nll = -logp.reshape(-1, V)[np.arange(target.size), target.reshape(-1)]
    total_nll += nll.sum(); total_tok += nll.size
print(f"ppl fused stateful path: {float(np.exp(total_nll/total_tok)):.4f}  (референс кернельного пути 11.7125)")

# --- 4. A/B скорость compiled ---
def bench(fuse, n=30, warm=8):
    qm.FUSE = fuse
    fn = mx.compile(model.forward_stateful)
    st = model.init_state(1)
    logits, st = fn(prompt, st, True)
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(warm):
        logits, st = fn(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = fn(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

pairs = [(bench(False), bench(True)) for _ in range(5)]
tu = np.median([p[0] for p in pairs]); tf = np.median([p[1] for p in pairs])
print(f"compiled unfused: {tu:6.2f} ms/tok | fused: {tf:6.2f} ms/tok | выигрыш {tu-tf:+.2f} ms")
print("пары:", " ".join(f"({u:.2f},{f:.2f})" for u, f in pairs))
