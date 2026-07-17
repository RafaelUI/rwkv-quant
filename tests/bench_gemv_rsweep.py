"""Свип R_PACKED (строк на тредгруппу) для v2 packed-GEMV: влияние на
незавершённые запросы к памяти / реюз x. Чередование в одном процессе."""
import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize_sparse_outlier
from rwkv_quant.formats.schema import QuantizedTensor, pack_int4
import rwkv_quant.backends.metal.quant_linear_v2 as qv2
from rwkv_quant.backends.metal.quant_linear_v2 import QuantLinearV2

torch.manual_seed(0); np.random.seed(0)
qv2.PACKED_V3 = False

def make(OUT, IN):
    w = torch.randn(OUT, IN) * torch.exp(torch.randn(OUT, 1))
    c, s, oi, ov = _real_quantize_sparse_outlier(w, 4, 0.02)
    return QuantizedTensor(key="t", group="proj", bits=4, shape=(OUT, IN),
                           codes=None, codes_packed=pack_int4(c), scale=s,
                           outlier_indices=oi, outlier_values=ov)

def bench_once(q, x, iters=200):
    for _ in range(3): mx.eval(q(x))
    mx.synchronize(); t0 = time.perf_counter()
    outs = [q(x) for _ in range(iters)]
    mx.eval(outs); mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

RS = [4, 8, 16]
a = mx.ones((2048, 2048), dtype=mx.float16)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.5: mx.eval(a @ a)

for OUT, IN in [(8192, 2048), (65536, 2048)]:
    q = QuantLinearV2(make(OUT, IN))
    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    res = {r: [] for r in RS}
    # паритет R=4/16 против R=8 (одна и та же q)
    qv2.R_PACKED = 8; qv2._packed_cache.clear(); yref = q(x); mx.eval(yref)
    for r in (4, 16):
        qv2.R_PACKED = r; qv2._packed_cache.clear()
        y = q(x); mx.eval(y)
        rel = float(mx.abs(y - yref).max() / (mx.abs(yref).max() + 1e-9))
        assert rel < 1e-5, f"R={r}: rel={rel:.2e}"
    for rnd in range(6):
        for r in RS:
            qv2.R_PACKED = r; qv2._packed_cache.clear()
            res[r].append(bench_once(q, x))
    nbytes = OUT * IN / 2 + int(0.02 * IN) * OUT * 4
    line = f"{OUT:>6}x{IN:<6} | " + " | ".join(
        f"R={r}: {statistics.median(res[r]):.3f}мс {nbytes/(statistics.median(res[r])*1e-3)/1e9:5.1f}GB/s" for r in RS)
    print(line)
qv2.R_PACKED = 8
print("готово")
