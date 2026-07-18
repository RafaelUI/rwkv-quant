"""Разделение: кернельная полоса vs launch-щели. cmix-матрицы чемпиона:
gw-int4 зависимая/независимая цепочка + v1 int4 (те же веса из pth)
зависимая/независимая. Чередование раундов."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal.quant_linear_v2 import QuantLinearV2
from rwkv_quant.calibration.group_config import QuantConfig

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
gw_lins = [l for b in model.blocks for l in (b.cmix.key, b.cmix.value)]

V1 = QuantConfig(proj=4, cmix=4, emb_head=4, w_lora=4, a_lora=4, v_lora=4,
                 g_lora=8, small=8, outlier_fracs={})
sd = torch.load(os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"),
                map_location="cpu")
v1_lins = []
for i in range(24):
    for nm in ("key", "value"):
        k = f"blocks.{i}.ffn.{nm}.weight"
        v1_lins.append(QuantLinearV2(quantize_tensor(k, sd[k], V1)))
del sd

def sz(lins):
    s = 0
    for l in lins:
        for a in vars(l).values():
            if isinstance(a, mx.array): s += a.size * a.itemsize
    return s / 1e6
MB_GW, MB_V1 = sz(gw_lins), sz(v1_lins)

x2048 = mx.array(np.random.randn(1, 2048).astype(np.float32))
x8192 = mx.array(np.random.randn(1, 8192).astype(np.float32))

def dep(lins):
    x = x2048
    for i in range(0, len(lins), 2):
        x = lins[i+1](mx.maximum(lins[i](x), 0.0))
    return [x]
def indep(lins):
    outs = []
    for i, l in enumerate(lins):
        outs.append(l(x2048 if i % 2 == 0 else x8192))
    return outs

def bench(fn, lins, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn(lins))
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn(lins)); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.median(ts)

cases = [("gw dep", dep, gw_lins, MB_GW), ("gw indep", indep, gw_lins, MB_GW),
         ("v1 dep", dep, v1_lins, MB_V1), ("v1 indep", indep, v1_lins, MB_V1)]
acc = {n: [] for n, *_ in cases}
for _ in range(5):
    for n, fn, lins, _mb in cases:
        acc[n].append(bench(fn, lins))
for n, fn, lins, MB in cases:
    t = np.median(acc[n])
    print(f"{n:9s} {t:7.3f} ms  {MB/t:6.1f} GB/s  ({MB:.0f}MB)")
