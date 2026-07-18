"""cmix key (2048->8192) vs value (8192->2048) отдельно, nb1, RB-свип, N=4."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
import rwkv_quant.backends.metal.quant_linear_gw as qlg

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
keys = [b.cmix.key for b in model.blocks]
vals = [b.cmix.value for b in model.blocks]
rng = np.random.default_rng(0)
xk = {N: mx.array(rng.standard_normal((N, 2048)).astype(np.float32)) for N in (1, 4)}
xv = {N: mx.array(rng.standard_normal((N, 8192)).astype(np.float32)) for N in (1, 4)}
for N in (1, 4): mx.eval(xk[N], xv[N])

def bench(fn, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

MBk = 24*9.7; MBv = 24*9.7  # ~ трафик весов на 24 слоя
res = {}
for r in range(5):
    for nm, lins, x in (("key", keys, xk), ("val", vals, xv)):
        res.setdefault((nm, "N1"), []).append(bench(lambda: [l(x[1]) for l in lins]))
        for rb in (2, 4, 8):
            qlg._RB_FOR_NN[4] = rb
            qlg._gw_kernel_cache = {k: v for k, v in qlg._gw_kernel_cache.items()
                                     if not (isinstance(k, tuple) and k[0] == "nb")}
            res.setdefault((nm, f"rb{rb}"), []).append(bench(lambda: [l(x[4]) for l in lins]))
for nm in ("key", "val"):
    b = float(np.median(res[(nm, "N1")]))
    line = f"{nm}: N=1 {b:6.2f}"
    for rb in (2, 4, 8):
        v = float(np.median(res[(nm, f"rb{rb}")]))
        line += f" | rb{rb} {v:6.2f} ({v/4:5.2f}/кол)"
    print(line)
