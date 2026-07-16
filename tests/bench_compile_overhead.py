"""Проверка: снимает ли mx.compile оверхед построения/диспатча графа на
цепочке из custom metal-кернелей (QuantLinear). Синтетический 'слой':
24 блока x (4 матмула 2048x2048 + 2 cmix 2048<->8192) ~ по числу кернелей
близко к реальному decode-шагу 1.5B (без WKV)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize
from rwkv_quant.formats.schema import QuantizedTensor
from rwkv_quant.backends.metal.quant_linear import QuantLinear

torch.manual_seed(0)
def make_ql(OUT, IN):
    codes, scale = _real_quantize(torch.randn(OUT, IN), 8)
    return QuantLinear(QuantizedTensor(key="b",group="proj",bits=8,shape=(OUT,IN),codes=codes,scale=scale))

N_LAYER = 24
layers = []
for _ in range(N_LAYER):
    layers.append(([make_ql(2048,2048) for _ in range(4)], make_ql(8192,2048), make_ql(2048,8192)))

def step(x):
    for projs, up, down in layers:
        h = x
        for p in projs:
            h = p(h)
        x = x + down(mx.maximum(up(h), 0))
    return x

x = mx.array(np.random.randn(1, 2048).astype(np.float32))

def bench(fn, n_warm=3, n_iter=20):
    for _ in range(n_warm): mx.eval(fn(x))
    mx.synchronize(); t0=time.perf_counter()
    for _ in range(n_iter): mx.eval(fn(x))
    mx.synchronize(); return (time.perf_counter()-t0)/n_iter*1e3

t_eager = bench(step)
step_c = mx.compile(step)
t_comp = bench(step_c)
print(f"eager:    {t_eager:.2f} ms/step")
print(f"compiled: {t_comp:.2f} ms/step  ({t_eager/t_comp:.2f}x)")
# сверка корректности
d = float(mx.abs(step(x) - step_c(x)).max())
print(f"max |diff| eager vs compiled: {d:.3e}")
