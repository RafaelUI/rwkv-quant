"""Кривая стоимости верификации для спекулятивного декодинга:
forward_stateful(T=k) чемпиона, k=1..16, амортизированный замер.
Если T=8 стоит ~1.2x T=1 -- спекулятивка окупается с запасом."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
rng = np.random.default_rng(0)

def bench_T(T, reps=20, warm=5):
    idx = mx.array(rng.integers(0, 65000, (1, T)).astype(np.int64))
    st = model.init_state(1)
    # тёплый state: прогнать 64 токена, чтобы не мерить холодный ноль
    warm_idx = mx.array(rng.integers(0, 65000, (1, 64)).astype(np.int64))
    lg, st = model.forward_stateful(warm_idx, st, last_only=True); mx.eval(lg)
    for _ in range(warm):
        lg, _ = model.forward_stateful(idx, st, last_only=False); mx.eval(lg)
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lg, _ = model.forward_stateful(idx, st, last_only=False); mx.eval(lg)
        mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

print(f"{'T':>3s} {'ms/проход':>10s} {'ms/ток':>8s} {'vs T=1':>7s}")
base = None
for T in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    ms = bench_T(T)
    if base is None: base = ms
    print(f"{T:3d} {ms:10.2f} {ms/T:8.2f} {ms/base:6.2f}x", flush=True)
