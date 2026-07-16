"""
Микробенч QuantLinear (наивный per-thread GEMV) против:
  - dense mx.matmul fp16 (потолок по скорости при полном трафике весов)
  - dense mx.matmul fp32
на реальных shapes 1.5B (d=2048), режим decode (N=1).

Цель: понять, сколько мы теряем на самом кернеле, а не на объёме памяти.
int8-веса тащат В ДВА РАЗА меньше байт, чем fp16 — если кернель был бы
memory-bound и оптимален, он обязан быть ~2x БЫСТРЕЕ fp16-матмула.
Всё, что медленнее, — потери на организации вычислений.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import mlx.core as mx

from rwkv_quant.formats.writer import _real_quantize, _real_quantize_sparse_outlier
from rwkv_quant.formats.schema import QuantizedTensor
from rwkv_quant.backends.metal.quant_linear import QuantLinear

torch.manual_seed(0)

SHAPES = [   # (out, in) — реальные 1.5B (d=2048)
    (2048, 2048),    # r/w/k/v/o proj
    (8192, 2048),    # cmix key
    (2048, 8192),    # cmix value
    (65536, 2048),   # head
]
N_WARM, N_ITER = 5, 30

def bench(fn):
    for _ in range(N_WARM):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1e3  # ms

print(f"{'shape':>14} | {'ql plain':>9} | {'ql spqr2%':>9} | {'fp16 mm':>8} | {'fp32 mm':>8} | ql/fp16")
for OUT, IN in SHAPES:
    w = torch.randn(OUT, IN)
    codes, scale = _real_quantize(w, 8)
    qt = QuantizedTensor(key="b", group="proj", bits=8, shape=(OUT, IN), codes=codes, scale=scale)
    ql = QuantLinear(qt)

    c2, s2, oi, ov = _real_quantize_sparse_outlier(w, 8, 0.02)
    qts = QuantizedTensor(key="b", group="proj", bits=8, shape=(OUT, IN),
                          codes=c2, scale=s2, outlier_indices=oi, outlier_values=ov)
    qls = QuantLinear(qts)

    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    w16 = mx.array(w.numpy()).astype(mx.float16)
    w32 = mx.array(w.numpy())
    x16 = x.astype(mx.float16)

    t_ql   = bench(lambda: ql(x))
    t_qls  = bench(lambda: qls(x))
    t_f16  = bench(lambda: x16 @ w16.T)
    t_f32  = bench(lambda: x @ w32.T)
    print(f"{OUT:>6}x{IN:<7} | {t_ql:8.3f}ms | {t_qls:8.3f}ms | {t_f16:7.3f}ms | {t_f32:7.3f}ms | {t_ql/t_f16:6.1f}x")
