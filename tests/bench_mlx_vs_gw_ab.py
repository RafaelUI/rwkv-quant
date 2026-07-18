"""A/B: наш gw sb6 GEMV-кернель vs нативный mx.quantized_matmul (MLX 0.32,
bits=4/5/6, gs=64) на ОДИНАКОВЫХ реальных формах чемпиона. Вопрос сессии:
почему MollySophia MLX INT6 end-to-end быстрее (84 GB/s эффективно) при
большем файле, тогда как наши чистые кернели 83-97 GB/s дают лишь 58 GB/s
end-to-end. Методология bench_kernel_clean.py: амортизированный синк,
A/B-чередование в одном процессе (закон N1), dep/indep цепочки.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
head = model.head                                # int5 65536x2048
cmix_key = model.blocks[0].cmix.key              # int4 8192x2048
tmix_lins = [l for b in model.blocks
             for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj)]

x = mx.array(np.random.randn(1, 2048).astype(np.float32)); mx.eval(x)
xh = x.astype(mx.float16); mx.eval(xh)


def gw_mb(l):
    s = (l.codes.size + l.qs.size + l.qm.size + l.d.size*2 + l.dm.size*2)
    if getattr(l, "qh", None) is not None and l.qh.size > 1: s += l.qh.size
    qh2 = getattr(l, "qh2", None)
    if qh2 is not None and qh2.size > 1: s += qh2.size
    return s / 1e6


class MlxLin:
    def __init__(self, w_f16, bits):
        self.wq, self.s, self.b = mx.quantize(w_f16, group_size=64, bits=bits)
        mx.eval(self.wq, self.s, self.b)
        self.bits = bits
        self.mb = (self.wq.size*4 + self.s.size*2 + self.b.size*2) / 1e6
    def __call__(self, x):
        return mx.quantized_matmul(x, self.wq, scales=self.s, biases=self.b,
                                   transpose=True, group_size=64, bits=self.bits)


print("построение MLX-двойников (деквант -> mx.quantize)...", flush=True)
head_mlx5 = MlxLin(head._dequant_w(), 5)
head_mlx6 = MlxLin(head._dequant_w(), 6)
cmix_mlx4 = MlxLin(cmix_key._dequant_w(), 4)
tmix_mlx5 = [MlxLin(l._dequant_w(), 5) for l in tmix_lins]
mx.eval(x)

MB_HEAD = gw_mb(head)
MB_CMIX = gw_mb(cmix_key)
MB_TMIX = sum(gw_mb(l) for l in tmix_lins)
MB_TMIX_MLX = sum(l.mb for l in tmix_mlx5)


def bench(fn, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def sync_only(): return [x + 1.0]

cases = {
    "gw head5 x8":   (lambda: [head(x) for _ in range(8)],        MB_HEAD*8),
    "mlx head5 x8":  (lambda: [head_mlx5(xh) for _ in range(8)],  head_mlx5.mb*8),
    "mlx head6 x8":  (lambda: [head_mlx6(xh) for _ in range(8)],  head_mlx6.mb*8),
    "gw cmixk4 x8":  (lambda: [cmix_key(x) for _ in range(8)],    MB_CMIX*8),
    "mlx cmixk4 x8": (lambda: [cmix_mlx4(xh) for _ in range(8)],  cmix_mlx4.mb*8),
    "gw tmix96 ind": (lambda: [l(x) for l in tmix_lins],          MB_TMIX),
    "mlx tmix96 ind":(lambda: [l(xh) for l in tmix_mlx5],         MB_TMIX_MLX),
}

def gw_dep():
    y = x
    for l in tmix_lins: y = l(y)
    return [y]
def mlx_dep():
    y = xh
    for l in tmix_mlx5: y = l(y)
    return [y]
cases["gw tmix96 dep"]  = (gw_dep,  MB_TMIX)
cases["mlx tmix96 dep"] = (mlx_dep, MB_TMIX_MLX)

acc = {n: [] for n in cases}; acc["sync"] = []
for r in range(5):
    for n, (fn, _) in cases.items():
        acc[n].append(bench(fn))
    acc["sync"].append(bench(sync_only, reps=30))

sync = float(np.median(acc["sync"]))
print(f"\nхост-синк: {sync:.3f} ms\n")
print(f"{'кейс':16s} {'ms':>8s} {'MB':>8s} {'GB/s':>7s}")
for n, (fn, mb) in cases.items():
    t = float(np.median(acc[n])) - sync
    print(f"{n:16s} {t:8.3f} {mb:8.1f} {mb/t:7.1f}")
