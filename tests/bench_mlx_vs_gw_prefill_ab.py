"""A/B GEMM-путь (T=1024): наш _dequant_w+matmul vs mx.quantized_matmul."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
cmix_key = model.blocks[0].cmix.key
tmix_r = model.blocks[0].tmix.r_proj

X = mx.array(np.random.randn(1024, 2048).astype(np.float32)); mx.eval(X)
Xh = X.astype(mx.float16); mx.eval(Xh)

class MlxLin:
    def __init__(self, w_f16, bits):
        self.wq, self.s, self.b = mx.quantize(w_f16, group_size=64, bits=bits)
        mx.eval(self.wq, self.s, self.b); self.bits = bits
    def __call__(self, x):
        return mx.quantized_matmul(x, self.wq, scales=self.s, biases=self.b,
                                   transpose=True, group_size=64, bits=self.bits)

cmix_mlx = MlxLin(cmix_key._dequant_w(), 4)
tmix_mlx = MlxLin(tmix_r._dequant_w(), 5)

def bench(fn, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

cases = {
    "gw  cmix GEMM x4": lambda: [cmix_key(X) for _ in range(4)],
    "mlx cmix GEMM x4": lambda: [cmix_mlx(Xh) for _ in range(4)],
    "gw  tmix GEMM x4": lambda: [tmix_r(X) for _ in range(4)],
    "mlx tmix GEMM x4": lambda: [tmix_mlx(Xh) for _ in range(4)],
}
acc = {n: [] for n in cases}
for r in range(5):
    for n, fn in cases.items(): acc[n].append(bench(fn))
for n in cases:
    t = float(np.median(acc[n]))/4
    # FLOPs: 2*T*IN*OUT
    if "cmix" in n: fl = 2*1024*2048*8192
    else: fl = 2*1024*2048*2048
    print(f"{n}: {t:7.3f} ms/op  {fl/t/1e9:7.0f} GFLOPS")
