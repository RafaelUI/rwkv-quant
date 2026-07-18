"""Кривая верификации через compiled model.step (mx.compile кэш по T)
vs raw forward_stateful. Плюс last_only-переключатель для оценки вклада
head на T позициях."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
rng = np.random.default_rng(0)

def bench(T, mode, reps=15, warm=6):
    idx = mx.array(rng.integers(0, 65000, (1, T)).astype(np.int64))
    warm_idx = mx.array(rng.integers(0, 65000, (1, 64)).astype(np.int64))
    st = model.init_state(1)
    lg, st = model.forward_stateful(warm_idx, st, last_only=True); mx.eval(lg)
    if mode == "step":
        fn = lambda: model.step(idx, st)
    elif mode == "raw":
        fn = lambda: model.forward_stateful(idx, st)
    else:  # raw_lastonly
        fn = lambda: model.forward_stateful(idx, st, last_only=True)
    for _ in range(warm):
        lg, _ = fn(); mx.eval(lg)
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lg, _ = fn(); mx.eval(lg)
        mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

print(f"{'T':>3s} {'raw':>8s} {'compiled':>9s} {'raw lastonly':>12s}")
for T in (1, 2, 4, 6, 8, 12):
    r = bench(T, "raw"); c = bench(T, "step"); lo = bench(T, "raw_lastonly")
    print(f"{T:3d} {r:8.2f} {c:9.2f} {lo:12.2f}", flush=True)
