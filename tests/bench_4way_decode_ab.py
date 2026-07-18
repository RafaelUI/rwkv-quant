"""A/B-interleaved decode-speed comparison, ONE process (methodology law
#1 in NEXT_SESSION.md: fanless thermal drift makes cross-process ms/tok
numbers incomparable -- 15.7->26ms drift observed under sustained load).
Loads all four real, deployable configs simultaneously and alternates
measurement rounds between them so any drift affects all four equally.

Configs:
  molly       -- MollySophia rwkv7-1.5B-g1g-mlx-6bit, native MLX affine
                 int6 (mx.quantized_matmul), tests/eval_molly_real.py loader
  compression -- this project's COMPRESSION preset, /tmp/champion_v2.rwkvq
  reduction   -- this project's REDUCTION v2 preset, /tmp/reduction_v2.rwkvq
  canonical6  -- plain per-row RTN int6 (no group scale/AW), real v1
                 QuantLinearV2 kernel, /tmp/canonical_int6.rwkvq

NOTE: molly is g1g, the other three are g1h (different BlinkDL release --
see chat caveat) -- fine for a same-process SPEED comparison (decode cost
depends on tensor shapes/quant scheme, which are architecturally
identical: hidden=2048, 24 layers, head_size=64 for both). It is NOT fine
to compare ppl this way (different fine-tunes) -- ppl is reported
separately per-config, already computed on the shared corpus in each
config's own eval script.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
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

print("loading canonical int6...")
canonical6 = QuantRWKV7(load_raw("/tmp/canonical_int6.rwkvq"))

MODELS = {"molly": molly, "compression": compression,
          "reduction": reduction, "canonical6": canonical6}

data = np.load(CORPUS) if CORPUS.endswith(".npy") else None
import torch
data = torch.load(CORPUS)[:1].numpy()
prompt64 = mx.array(data[0:1, :64].astype(np.int32))

states, toks = {}, {}
for name, m in MODELS.items():
    st = m.init_state(1)
    logits, st = m.forward_stateful(prompt64, st, last_only=True)
    states[name] = st
    toks[name] = mx.argmax(logits[:, -1], axis=-1)

# warmup every model (compile caches etc.)
for _ in range(8):
    for name, m in MODELS.items():
        logits, states[name] = m.step(toks[name][None], states[name])
        toks[name] = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(*[toks[n] for n in MODELS])

# A/B rounds: interleave, R rounds of N steps each, report per-round times
R, N = 4, 32
times = {name: [] for name in MODELS}
for r in range(R):
    for name, m in MODELS.items():
        t0 = time.time()
        for _ in range(N):
            logits, states[name] = m.step(toks[name][None], states[name])
            toks[name] = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(toks[name])
        dt = (time.time() - t0) / N * 1000
        times[name].append(dt)
        print(f"  round {r} {name:12s} {dt:.2f} ms/tok")

print()
for name in MODELS:
    ts = times[name]
    print(f"{name:12s} mean={np.mean(ts):.2f}  min={np.min(ts):.2f}  max={np.max(ts):.2f}  ms/tok")
