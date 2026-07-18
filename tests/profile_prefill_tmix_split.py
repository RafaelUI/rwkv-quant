"""Как profile_prefill_dequant_split.py, но для tmix r/k/v/o-проекций --
проверить, тоже ли они близки к compute ceiling (2733 GFLOPS, из
bench_compute_ceiling.py сессии 19.07-3) или там другая арифметическая
интенсивность/узкое место."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
model = qm.QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
tm = model.blocks[5].tmix
lins = {"r_proj": tm.r_proj, "k_proj": tm.k_proj, "v_proj": tm.v_proj, "o_proj": tm.o_proj}
for name, lin in lins.items():
    print(f"{name}: out={lin.out_features} in={lin.in_features} xbits={getattr(lin,'xbits','?')}")


def spin(sec=2.0):
    a = mx.ones((2048, 2048), dtype=mx.float16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        mx.eval(a @ a)


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


spin()
print(f"\n=== T={T} ===")
for name, lin in lins.items():
    x = mx.array(np.random.randn(T, lin.in_features).astype(np.float32))
    R = 4
    dq_t, mm_t, full_t = [], [], []
    for _ in range(R):
        t_dq = bench(lambda: lin._dequant_w())
        w = lin._dequant_w(); mx.eval(w)
        t_mm = bench(lambda: mx.matmul(x.astype(mx.float16), w.T))
        t_full = bench(lambda: lin(x))
        dq_t.append(t_dq); mm_t.append(t_mm); full_t.append(t_full)
    dq, mm, fu = np.median(dq_t), np.median(mm_t), np.median(full_t)
    flops = 2 * T * lin.in_features * lin.out_features
    gflops = flops / (mm * 1e-3) / 1e9
    print(f"{name:8s} dequant={dq:6.2f}ms matmul={mm:6.2f}ms full={fu:6.2f}ms "
          f"dq_share={dq/(dq+mm)*100:4.1f}%  matmul_GFLOPS={gflops:7.0f}")
