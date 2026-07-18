"""Изоляция: сколько времени в GEMM-префилле (GwQuantLinear.__call__, N>=16)
уходит на _dequant_w() (материализация fp16 [OUT,IN]) против самого
mx.matmul. Цель -- понять, стоит ли писать fused dequant+GEMM Metal-кернель
(избежать материализации транзиентного fp16-буфера) или узкое место -- сам
матмул (тогда фьюз бесполезен, дело в FLOPs/roofline).

Берём реальные cmix key/value тензоры чемпиона (OUT=8192,IN=2048 и
OUT=2048,IN=8192, самая тяжёлая группа). A/B-чередование в одном процессе.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
model = qm.QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))

# реальные cmix-слои чемпиона, слой 5 (произвольный, подальше от layer 0)
key_lin = model.blocks[5].cmix.key      # GwQuantLinear OUT=8192? IN=2048 (расширение)
val_lin = model.blocks[5].cmix.value    # GwQuantLinear OUT=2048, IN=8192 (сжатие)
print(f"key: out={key_lin.out_features} in={key_lin.in_features} xbits={key_lin.xbits}")
print(f"value: out={val_lin.out_features} in={val_lin.in_features} xbits={val_lin.xbits}")


def spin(sec=2.0):
    a = mx.ones((2048, 2048), dtype=mx.float16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        mx.eval(a @ a)


def bench(fn, n=8, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


spin()

x_key = mx.array(np.random.randn(T, key_lin.in_features).astype(np.float32))
x_val = mx.array(np.random.randn(T, val_lin.in_features).astype(np.float32))

print(f"\n=== T={T}, R раундов A/B-чередование ===")
for name, lin, x in [("key (2048->8192)", key_lin, x_key), ("value (8192->2048)", val_lin, x_val)]:
    R = 5
    dequant_times, matmul_times, full_times = [], [], []
    for _ in range(R):
        t_dq = bench(lambda: lin._dequant_w(), n=5, warm=2)
        w = lin._dequant_w()
        mx.eval(w)
        t_mm = bench(lambda: mx.matmul(x.astype(mx.float16), w.T), n=5, warm=2)
        t_full = bench(lambda: lin(x), n=5, warm=2)
        dequant_times.append(t_dq); matmul_times.append(t_mm); full_times.append(t_full)
    dq, mm, fu = np.median(dequant_times), np.median(matmul_times), np.median(full_times)
    print(f"{name:22s} dequant={dq:7.2f} ms  matmul={mm:7.2f} ms  "
          f"sum={dq+mm:7.2f} ms  full(__call__)={fu:7.2f} ms  "
          f"dequant_share={dq/(dq+mm)*100:.1f}%")
