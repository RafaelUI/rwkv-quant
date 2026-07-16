"""Доля WKV в decode-шаге: замер wkv7_infer (T=CHUNK ради одного токена,
как в _wkv_stateful при T=1) на конфигурации 1.5B (24 слоя, H=32, D=64)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("RWKV_METAL_PATH", os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.environ["RWKV_METAL_PATH"])
import numpy as np, mlx.core as mx
from rwkv_metal.kernel.wkv7 import wkv7_infer, CHUNK

B, H, D, NL = 1, 32, 64, 24
r,w,k,v,a,b = [mx.array(np.random.randn(B,CHUNK,H,D).astype(np.float32))*0.1 for _ in range(6)]
w = mx.array(np.exp(-np.exp(np.random.randn(B,CHUNK,H,D).astype(np.float32))))
h = mx.zeros((B,H,D,D))

def step():
    out = []
    hh = h
    for _ in range(NL):
        o, hh = wkv7_infer(r,w,k,v,a,b,hh)
        out.append(o)
    return out, hh

for _ in range(3): mx.eval(step())
mx.synchronize(); t0=time.perf_counter()
N=20
for _ in range(N): mx.eval(step())
mx.synchronize()
t = (time.perf_counter()-t0)/N*1e3
print(f"CHUNK={CHUNK}: 24x wkv7_infer = {t:.2f} ms/step  ({t/NL:.3f} ms/слой)")
