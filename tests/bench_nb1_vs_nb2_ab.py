"""A/B nb1 vs nb2 в ОДНОМ процессе (закон N1), чередование по раундам."""
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
xk = {N: mx.array(rng.standard_normal((N, 2048)).astype(np.float32)) for N in (1, 4)}
xv = {N: mx.array(rng.standard_normal((N, 8192)).astype(np.float32)) for N in (1, 4)}
for N in (1, 4): mx.eval(xk[N], xv[N])

def cmix_run(N):
    outs = []
    for b in model.blocks:
        outs.append(b.cmix.key(xk[N])); outs.append(b.cmix.value(xv[N]))
    return outs
def tmix_run(N):
    return [l(xk[N]) for l in tmix]
def head_run(N):
    return [model.head(xk[N])]

def bench(fn, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

cases = {"tmix4": lambda: tmix_run(4), "cmix4": lambda: cmix_run(4), "head4": lambda: head_run(4)}
acc = {(v, n): [] for v in ("nb1", "nb2") for n in cases}
base = {n: [] for n in ("tmix1", "cmix1", "head1")}
for r in range(5):
    base["tmix1"].append(bench(lambda: tmix_run(1)))
    base["cmix1"].append(bench(lambda: cmix_run(1)))
    base["head1"].append(bench(lambda: head_run(1)))
    for v in ("nb1", "nb2"):
        qlg.NB_V2 = (v == "nb2")
        for n, fn in cases.items():
            acc[(v, n)].append(bench(fn))
for n in ("tmix", "cmix", "head"):
    b = float(np.median(base[n+"1"]))
    v1 = float(np.median(acc[("nb1", n+"4")])); v2 = float(np.median(acc[("nb2", n+"4")]))
    print(f"{n}: N=1 {b:6.2f} | nb1 N=4 {v1:6.2f} ({v1/4:5.2f}/кол) | nb2 N=4 {v2:6.2f} ({v2/4:5.2f}/кол)")
