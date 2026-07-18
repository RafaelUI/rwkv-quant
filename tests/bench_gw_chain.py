"""Агрегатный GB/s gw-кернеля на боевых цепочках чемпиона: один eval на
целый проход (48 cmix-GEMV / 96 tmix-GEMV / head), launch амортизирован.
Чередование раундов cmix/tmix/head (закон №1)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))

cmix_pairs = [(b.cmix.key, b.cmix.value) for b in model.blocks]
tmix_lins = [l for b in model.blocks
             for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj)]
head = model.head

def mb(lin):
    if isinstance(lin, GwQuantLinear):
        t = 0
        for a in (lin.qs, lin.qm, lin.d, lin.dm, getattr(lin, "qh", None), lin.codes if hasattr(lin, "codes") else None):
            if isinstance(a, mx.array): t += a.size * a.itemsize
        return t / 1e6
    return 0.0

def total_mb(lins):
    s = 0.0
    for l in lins:
        m = mb(l)
        if m == 0.0:  # не gw -- посчитать по атрибутам общего вида
            for a in vars(l).values():
                if isinstance(a, mx.array): s += a.size * a.itemsize / 1e6
        else: s += m
    return s

MB_CMIX = total_mb([l for p in cmix_pairs for l in p])
MB_TMIX = total_mb(tmix_lins)
MB_HEAD = total_mb([head])
print(f"traffic: cmix {MB_CMIX:.0f}MB, tmix {MB_TMIX:.0f}MB, head {MB_HEAD:.0f}MB")

x2048 = mx.array(np.random.randn(1, 2048).astype(np.float32))

def pass_cmix():
    x = x2048
    for k, v in cmix_pairs: x = v(mx.maximum(k(x), 0.0))
    return x
def pass_tmix():
    x = x2048
    for l in tmix_lins: x = l(x)
    return x
def pass_head(): return head(x2048)

def bench(fn, reps=12, warm=3):
    for _ in range(warm): mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.median(ts)

res = {"cmix": [], "tmix": [], "head": []}
for _ in range(5):
    res["cmix"].append(bench(pass_cmix))
    res["tmix"].append(bench(pass_tmix))
    res["head"].append(bench(pass_head))

for name, MB, nlin in (("cmix", MB_CMIX, 48), ("tmix", MB_TMIX, 96), ("head", MB_HEAD, 1)):
    t = np.median(res[name])
    print(f"{name:5s} {t:7.3f} ms/pass  {MB/t:6.1f} GB/s  ({nlin} GEMV, {t/nlin*1000:5.1f} us/GEMV)")
