"""Префилл 1.5B COMPRESSION (packed): GEMV vs GEMM-путь QuantLinearV2.
A/B в ОДНОМ процессе с прогревом (методология NEXT_SESSION №4/№4b).
Сырой forward_stateful (не model.step): compile-граф кешируется по shapes
и не увидел бы подмену GEMM_MIN_BATCH."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_v2 as qv2

T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ckpt = load_raw("/tmp/compression_packed.rwkvq")
model = qm.QuantRWKV7(ckpt)
idx = mx.array(np.random.randint(0, 65000, (1, T)).astype(np.int64))

def spin(sec=2.0):
    a = mx.ones((2048, 2048), dtype=mx.float16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec: mx.eval(a @ a)

def _flat(st): return [s for x in st for s in x if s is not None]

def bench(iters=8):
    for _ in range(2):
        logits, st = model.forward_stateful(idx, model.init_state(1), last_only=True)
        mx.eval(logits, *_flat(st))
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        logits, st = model.forward_stateful(idx, model.init_state(1), last_only=True)
        mx.eval(logits, *_flat(st))
    mx.synchronize(); return (time.perf_counter() - t0) / iters * 1e3

spin()
saved = qv2.GEMM_MIN_BATCH
qv2.GEMM_MIN_BATCH = 10**9
t_gemv = bench()
qv2.GEMM_MIN_BATCH = saved
t_gemm = bench()
print(f"T={T}: GEMV {t_gemv:.0f} мс ({t_gemv/T:.2f} мс/ток) -> GEMM {t_gemm:.0f} мс "
      f"({t_gemm/T:.2f} мс/ток), speedup {t_gemv/t_gemm:.2f}x")
