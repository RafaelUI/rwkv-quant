"""Изолированная цель под Metal System Trace / GPU capture.

    python tests/profile_target.py prefill [T=1024] [seconds=40]
    python tests/profile_target.py decode [seconds=40]        # mx.compile step
    python tests/profile_target.py decode_eager [seconds=40]  # сырой forward_stateful

Печатает PID и фазы. Захват делать ТОЛЬКО после строки STEADY -- до неё
прогрев (DVFS) и компиляции шейдеров, они зашумят профиль. Вопросы к трейсу:
prefill T=1024 -- что осталось после GEMM-пути (доля WKV-chunked,
elementwise, пузыри между диспатчами = CPU-bound Python-цикла);
decode -- доля шести token-shift lerp'ов после compile (резерв фьюза №4b)
и счётчик компиляций шейдеров в стационаре (должен быть 0)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

mode = sys.argv[1] if len(sys.argv) > 1 else "prefill"
if mode == "prefill":
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
else:
    T, seconds = 1, (float(sys.argv[2]) if len(sys.argv) > 2 else 40.0)

print(f"PID {os.getpid()}  mode={mode} T={T} steady={seconds:.0f}s", flush=True)
model = QuantRWKV7(load_raw("/tmp/compression_packed.rwkvq"))

def _flat(st): return [s for x in st for s in x if s is not None]

if mode == "prefill":
    idx = mx.array(np.random.randint(0, 65000, (1, T)).astype(np.int64))
    def once():
        logits, st = model.forward_stateful(idx, model.init_state(1), last_only=True)
        mx.eval(logits, *_flat(st))
else:
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
per = dt / n * 1e3
print(f"DONE: {n} итераций, {per:.2f} мс/итер" + (f" ({per/T:.3f} мс/ток)" if mode == "prefill" else ""), flush=True)
