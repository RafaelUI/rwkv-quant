"""Как profile_target.py, но: чемпион-чекпоинт (/tmp/champion_v2.rwkvq) +
настраиваемый FUSE, для A/B command-buffer-count через xctrace/Metal System
Trace (см. tests/xctrace_cmdbuf_ab.sh).

    python tests/profile_target_champion.py decode [seconds=40] [fuse=1]

Печатает PID и STEADY. Захват -- строго ПОСЛЕ строки STEADY.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

mode = sys.argv[1] if len(sys.argv) > 1 else "decode"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
fuse = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
qm.FUSE = fuse

print(f"PID {os.getpid()}  mode={mode} FUSE={fuse} steady={seconds:.0f}s", flush=True)
model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))

def _flat(st): return [s for x in st for s in x if s is not None]

fwd = model.step if mode == "decode" else model.forward_stateful
state = model.init_state(1)
tok = mx.array(np.array([[123]], dtype=np.int64))
def once():
    global state, tok
    logits, state = fwd(tok, state)
    tok = mx.argmax(logits[:, -1, :], axis=-1)[None]
    mx.eval(tok, *_flat(state))

print("warmup...", flush=True)
a = mx.ones((2048, 2048), dtype=mx.float16)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 2.0: mx.eval(a @ a)
for _ in range(5): once()
mx.synchronize()

print(f"STEADY (можно захватывать, {seconds:.0f}s)", flush=True)
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < seconds:
    once(); n += 1
mx.synchronize()
dt = time.perf_counter() - t0
print(f"DONE: {n} итераций, {dt/n*1e3:.4f} мс/ток, dt={dt:.4f}s", flush=True)
