"""A/B-interleaved decode AND prefill speed, one process, four configs
(19.07-10 follow-up: canonical INT6 swapped for canonical INT4 -- honestly
nibble-packed, so it's not the odd-one-out on size anymore; prefill added,
was missing from the first pass).

Configs: molly (MollySophia g1g mlx-6bit, native MLX affine int6),
canonical4 (plain per-row RTN int4, real nibble packing),
compression (ours, gw sb6 mixed 4/5-bit), reduction (ours, gw sb6 int6).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import mlx.core as mx

from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from tests.eval_molly_real import MollyRWKV7, MODEL_DIR

CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")

print("loading molly...")
w = mx.load(f"{MODEL_DIR}/model.safetensors")
molly = MollyRWKV7(w)
del w

print("loading compression...")
compression = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))

print("loading reduction v2...")
reduction = QuantRWKV7(load_raw("/tmp/reduction_v2.rwkvq"))

print("loading canonical int4...")
canonical4 = QuantRWKV7(load_raw("/tmp/canonical_int4.rwkvq"))

MODELS = {"molly": molly, "compression": compression,
          "reduction": reduction, "canonical4": canonical4}

data = torch.load(CORPUS)[:1].numpy()
prompt64 = mx.array(data[0:1, :64].astype(np.int32))

# ============================== DECODE ==============================
states, toks = {}, {}
for name, m in MODELS.items():
    st = m.init_state(1)
    logits, st = m.forward_stateful(prompt64, st, last_only=True)
    states[name] = st
    toks[name] = mx.argmax(logits[:, -1], axis=-1)

for _ in range(8):
    for name, m in MODELS.items():
        logits, states[name] = m.step(toks[name][None], states[name])
        toks[name] = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(*[toks[n] for n in MODELS])

R, N = 4, 32
dec_times = {name: [] for name in MODELS}
print("\n=== DECODE (single-token, A/B rounds) ===")
for r in range(R):
    for name, m in MODELS.items():
        t0 = time.time()
        for _ in range(N):
            logits, states[name] = m.step(toks[name][None], states[name])
            toks[name] = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(toks[name])
        dt = (time.time() - t0) / N * 1000
        dec_times[name].append(dt)
        print(f"  round {r} {name:12s} {dt:.2f} ms/tok")

print()
for name in MODELS:
    ts = dec_times[name]
    print(f"decode {name:12s} mean={np.mean(ts):.2f}  min={np.min(ts):.2f}  max={np.max(ts):.2f}  ms/tok")

# ============================== PREFILL ==============================
# T=1024 random tokens (content doesn't matter for a speed benchmark,
# only shape does -- same convention as profile_prefill_v2.py).
T = 1024
np.random.seed(0)
idx1024 = mx.array(np.random.randint(0, 65000, (1, T)).astype(np.int64))


def _flat(st):
    return [s for x in st for s in x if s is not None]


def timed_prefill(m, n=3, warm=2):
    for _ in range(warm):
        logits, st = m.forward_stateful(idx1024, m.init_state(1), last_only=True)
        mx.eval(logits, *_flat(st))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = m.forward_stateful(idx1024, m.init_state(1), last_only=True)
        mx.eval(logits, *_flat(st))
    mx.synchronize()
    return (time.perf_counter() - t0) / n  # seconds per full T=1024 prefill


R_PF = 3
pf_times = {name: [] for name in MODELS}
print(f"\n=== PREFILL (T={T}, A/B rounds) ===")
for r in range(R_PF):
    for name, m in MODELS.items():
        dt = timed_prefill(m, n=3, warm=1 if r > 0 else 2)
        ms_per_tok = dt / T * 1000
        toks_per_s = T / dt
        pf_times[name].append(toks_per_s)
        print(f"  round {r} {name:12s} {dt*1000:8.1f} ms total  "
              f"{ms_per_tok:.4f} ms/tok  {toks_per_s:8.1f} tok/s")

print()
for name in MODELS:
    ts = pf_times[name]
    print(f"prefill {name:12s} mean={np.mean(ts):8.1f}  min={np.min(ts):8.1f}  max={np.max(ts):8.1f}  tok/s")
