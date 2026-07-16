"""Отделяем оверхед диспатча/eval от собственно compute.
1) 'пол': тривиальная операция (add на 16 float) через mx.eval на каждой итерации.
2) те же GEMV, но 30 итераций в очередь -> ОДИН mx.eval: если время/итерацию
   падает в разы, узкое место -- overhead per-eval, не кернель."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize
from rwkv_quant.formats.schema import QuantizedTensor
from rwkv_quant.backends.metal.quant_linear import QuantLinear

torch.manual_seed(0)
N_WARM, N_ITER = 5, 30

def bench_evalstep(fn):
    for _ in range(N_WARM): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(N_ITER): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter()-t0)/N_ITER*1e3

def bench_chained(fn, x0):
    # y = fn(fn(...)) невозможно (shape), так что просто копим список и eval разом
    for _ in range(N_WARM): mx.eval(fn(x0))
    mx.synchronize(); t0 = time.perf_counter()
    outs = [fn(x0) for _ in range(N_ITER)]
    mx.eval(outs)
    mx.synchronize(); return (time.perf_counter()-t0)/N_ITER*1e3

a = mx.array(np.random.randn(16).astype(np.float32))
b = mx.array(np.random.randn(16).astype(np.float32))
print(f"floor (16-elem add, eval/step): {bench_evalstep(lambda: a+b):.3f}ms")
outs=[a+b for _ in range(N_ITER)]; mx.synchronize()
t0=time.perf_counter(); outs=[a+b for _ in range(N_ITER)]; mx.eval(outs); mx.synchronize()
print(f"floor (16-elem add, batched eval): {(time.perf_counter()-t0)/N_ITER*1e3:.3f}ms")

for OUT, IN in [(2048,2048),(8192,2048),(65536,2048)]:
    w = torch.randn(OUT, IN)
    codes, scale = _real_quantize(w, 8)
    ql = QuantLinear(QuantizedTensor(key="b",group="proj",bits=8,shape=(OUT,IN),codes=codes,scale=scale))
    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    w16 = mx.array(w.numpy()).astype(mx.float16); x16 = x.astype(mx.float16)
    print(f"{OUT}x{IN}: ql eval/step {bench_evalstep(lambda: ql(x)):.3f}ms | "
          f"ql batched {bench_chained(ql, x):.3f}ms | "
          f"fp16 batched {bench_chained(lambda t: t.astype(mx.float16) @ w16.T, x):.3f}ms")
