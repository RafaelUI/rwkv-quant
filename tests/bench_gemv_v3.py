"""A/B v2 (uchar4) vs v3 (uint4) packed-GEMV, N=1. Чередование раундов в
ОДНОМ процессе: тепловой/своповый дрейф машины бьёт по обоим вариантам
поровну (урок сессии №4g: последовательные замеры на деградирующей машине
дали 15.7->26.3 мс на неизменном коде). Медианы по 6 раундам."""
import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize_sparse_outlier
from rwkv_quant.formats.schema import QuantizedTensor, pack_int4
import rwkv_quant.backends.metal.quant_linear_v2 as qv2
from rwkv_quant.backends.metal.quant_linear_v2 import QuantLinearV2

torch.manual_seed(0); np.random.seed(0)

def make(OUT, IN):
    w = torch.randn(OUT, IN) * torch.exp(torch.randn(OUT, 1))
    c, s, oi, ov = _real_quantize_sparse_outlier(w, 4, 0.02)
    return QuantizedTensor(key="t", group="proj", bits=4, shape=(OUT, IN),
                           codes=None, codes_packed=pack_int4(c), scale=s,
                           outlier_indices=oi, outlier_values=ov)

print("== паритет v3 vs v2 (N=1 и N=4) ==")
for OUT, IN in [(2048, 2048), (8192, 2048), (2048, 8192), (65536, 2048), (768, 3072)]:
    q = QuantLinearV2(make(OUT, IN))
    for N in (1, 4):
        x = mx.array(np.random.randn(N, IN).astype(np.float32))
        qv2.PACKED_V3 = False; y2 = q(x); mx.eval(y2)
        qv2.PACKED_V3 = True;  y3 = q(x); mx.eval(y3)
        rel = float(mx.abs(y2 - y3).max() / (mx.abs(y2).max() + 1e-9))
        tag = "v3" if (IN % 32 == 0) else "fallback->v2"
        assert rel < 1e-5, f"{OUT}x{IN} N={N}: rel={rel:.2e}"
        print(f"OK  {OUT}x{IN} N={N} rel {rel:.2e} ({tag})")

def bench_once(q, x, iters):
    # батчевый eval: pos-eval КАЖДОЙ итерации меряет пол диспатча (~0.2мс,
    # см. bench_dispatch_floor.py), не кернель. GEMV не держит больших
    # промежутков -- батчить безопасно (в отличие от деквант-графов №4c).
    for _ in range(3): mx.eval(q(x))
    mx.synchronize(); t0 = time.perf_counter()
    outs = [q(x) for _ in range(iters)]
    mx.eval(outs)
    mx.synchronize(); return (time.perf_counter() - t0) / iters * 1e3

print("\n== бенч N=1, медианы 6 чередующихся раундов ==")
a = mx.ones((2048, 2048), dtype=mx.float16)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 1.5: mx.eval(a @ a)

print(f"{'shape':>14} | {'v2 мс':>8} | {'v3 мс':>8} | {'v2 GB/s':>7} | {'v3 GB/s':>7} | v2/v3")
for OUT, IN in [(2048, 2048), (8192, 2048), (2048, 8192), (65536, 2048)]:
    q = QuantLinearV2(make(OUT, IN))
    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    iters = 100 if OUT >= 65536 else 300
    t2, t3 = [], []
    for r in range(6):
        qv2.PACKED_V3 = False; t2.append(bench_once(q, x, iters))
        qv2.PACKED_V3 = True;  t3.append(bench_once(q, x, iters))
    m2, m3 = statistics.median(t2), statistics.median(t3)
    nbytes = OUT * IN / 2 + int(0.02 * IN) * OUT * (2 + 2)  # codes + outliers(cols u16 + vals bf16)
    gb2, gb3 = nbytes / (m2 * 1e-3) / 1e9, nbytes / (m3 * 1e-3) / 1e9
    print(f"{OUT:>6}x{IN:<7} | {m2:8.3f} | {m3:8.3f} | {gb2:7.1f} | {gb3:7.1f} | {m2/m3:5.2f}x")
qv2.PACKED_V3 = True
print("\nготово")
