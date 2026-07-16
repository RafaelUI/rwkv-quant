"""Численная эквивалентность QuantLinearV2 против v1 (эталон, провалидирован
против fp32/bf16 в test_quant_linear_metal.py) + бенч v1 vs v2 vs fp16."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize, _real_quantize_sparse_outlier
from rwkv_quant.formats.schema import QuantizedTensor
from rwkv_quant.backends.metal.quant_linear import QuantLinear
from rwkv_quant.backends.metal.quant_linear_v2 import QuantLinearV2

torch.manual_seed(0); np.random.seed(0)

def make(OUT, IN, bits, spqr):
    w = torch.randn(OUT, IN) * torch.exp(torch.randn(OUT, 1))
    if spqr:
        c, s, oi, ov = _real_quantize_sparse_outlier(w, bits, 0.02)
        return QuantizedTensor(key="t", group="proj", bits=bits, shape=(OUT,IN),
                               codes=c, scale=s, outlier_indices=oi, outlier_values=ov)
    c, s = _real_quantize(w, bits)
    return QuantizedTensor(key="t", group="proj", bits=bits, shape=(OUT,IN), codes=c, scale=s)

print("== эквивалентность v2 vs v1 ==")
ok = True
for OUT, IN in [(2048,2048),(8192,2048),(2048,8192),(768,3072),(65536,2048)]:
    for bits in (4, 8):
        for spqr in (False, True):
            qt = make(OUT, IN, bits, spqr)
            q1, q2 = QuantLinear(qt), QuantLinearV2(qt)
            for N in (1, 4):
                x = mx.array(np.random.randn(N, IN).astype(np.float32))
                y1, y2 = q1(x), q2(x)
                rel = float(mx.abs(y1-y2).max() / (mx.abs(y1).max()+1e-9))
                status = "OK " if rel < 1e-5 else "FAIL"
                if rel >= 1e-5: ok = False
                if N == 1:
                    print(f"{status} {OUT}x{IN} bits={bits} spqr={spqr}: max rel {rel:.2e}")
assert ok, "v2 численно разошёлся с v1"

print("\n== бенч (batched eval, N=1) ==")
N_WARM, N_ITER = 5, 50
def bench(fn, x):
    for _ in range(N_WARM): mx.eval(fn(x))
    mx.synchronize(); t0=time.perf_counter()
    outs=[fn(x) for _ in range(N_ITER)]; mx.eval(outs)
    mx.synchronize(); return (time.perf_counter()-t0)/N_ITER*1e3

print(f"{'shape':>14} | {'v1':>8} | {'v2':>8} | {'v2 spqr':>8} | {'fp16':>8} | v1/v2")
for OUT, IN in [(2048,2048),(8192,2048),(2048,8192),(65536,2048)]:
    qt  = make(OUT, IN, 8, False)
    qts = make(OUT, IN, 8, True)
    q1, q2, q2s = QuantLinear(qt), QuantLinearV2(qt), QuantLinearV2(qts)
    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    w16 = mx.array(torch.randn(OUT, IN).numpy()).astype(mx.float16)
    t1, t2, t2s = bench(q1, x), bench(q2, x), bench(q2s, x)
    tf = bench(lambda t: t.astype(mx.float16) @ w16.T, x)
    print(f"{OUT:>6}x{IN:<7} | {t1:7.3f} | {t2:7.3f} | {t2s:7.3f} | {tf:7.3f} | {t1/t2:5.2f}x")
