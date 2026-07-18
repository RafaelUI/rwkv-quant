"""Свип N-батчевого кернеля: tmix-цепочка 96 GEMV indep, N x RB.
Метрика: мс на КОЛОНКУ (= на токен верификации) -- чем ниже, тем лучше;
N=1 старый кернель = якорь."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
import rwkv_quant.backends.metal.quant_linear_gw as qlg

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
tmix = [l for b in model.blocks
        for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj)]
rng = np.random.default_rng(0)

def bench(fn, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

x1 = mx.array(rng.standard_normal((1, 2048)).astype(np.float32)); mx.eval(x1)
base = bench(lambda: [l(x1) for l in tmix])
print(f"N= 1 (старый):        {base:7.2f} ms  {base:6.2f} ms/кол")

for N in (2, 4, 6, 8, 12):
    xN = mx.array(rng.standard_normal((N, 2048)).astype(np.float32)); mx.eval(xN)
    for rb in (2, 4, 8):
        qlg.RB_NB = rb
        qlg._gw_kernel_cache = {k: v for k, v in qlg._gw_kernel_cache.items()
                                 if not (isinstance(k, tuple) and k and k[0] == "nb")}
        ms = bench(lambda: [l(xN) for l in tmix])
        print(f"N={N:2d} RB={rb}:            {ms:7.2f} ms  {ms/N:6.2f} ms/кол", flush=True)
