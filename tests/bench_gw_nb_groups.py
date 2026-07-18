"""Кернельный бюджет T=4: tmix/cmix/head отдельно, N=1 vs N=4 (чанк),
чтобы вычислить GEMV-часть verify-прохода и остаток на не-GEMV."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
tmix = [l for b in model.blocks
        for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj)]
cmix = [l for b in model.blocks for l in (b.cmix.key, b.cmix.value)]
head = [model.head]
rng = np.random.default_rng(0)

def bench(fn, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

for name, lins in (("tmix96", tmix), ("cmix48", cmix), ("head", head)):
    IN = lins[0].in_features
    xs = {N: mx.array(rng.standard_normal((N, IN)).astype(np.float32)) for N in (1, 4)}
    # cmix: key 2048->8192, value 8192->2048 -- вход разной ширины, бенчим парами
    if name == "cmix48":
        xk = {N: mx.array(rng.standard_normal((N, 2048)).astype(np.float32)) for N in (1, 4)}
        xv = {N: mx.array(rng.standard_normal((N, 8192)).astype(np.float32)) for N in (1, 4)}
        for N in (1, 4):
            mx.eval(xk[N], xv[N])
            ms = bench(lambda: [ (b.cmix.key(xk[N]), b.cmix.value(xv[N])) and None or b.cmix.key(xk[N]) for b in model.blocks ][:1] or None) if False else None
        # проще: явные списки
        def run(N):
            outs = []
            for b in model.blocks:
                outs.append(b.cmix.key(xk[N])); outs.append(b.cmix.value(xv[N]))
            return outs
        for N in (1, 4):
            ms = bench(lambda N=N: run(N))
            print(f"{name} N={N}: {ms:7.2f} ms  ({ms/N:6.2f} мс/кол)", flush=True)
        continue
    for N in (1, 4):
        mx.eval(xs[N])
        ms = bench(lambda N=N: [l(xs[N]) for l in lins])
        print(f"{name} N={N}: {ms:7.2f} ms  ({ms/N:6.2f} мс/кол)", flush=True)
