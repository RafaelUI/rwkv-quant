"""
Синтетический тест backends/metal/quant_linear.py: сверяем результат
QuantLinear (int8 codes + per-row scale + SpQR, матмул напрямую на Metal)
против formats.reader._dequantize_one (разворачивание в bf16 + обычный
torch matmul) -- на случайных весах, с двумя режимами: plain RTN и
RTN+SpQR sparse outlier.

Два эталона:
  - fp32-reference: та же формула dequant (codes*scale + outlier overwrite),
    но БЕЗ каста в bf16 -- изолирует "кернель верно считает codes/scale/outlier"
    от "reader.py дополнительно теряет точность через bf16 storage".
  - bf16-reference (реальный _dequantize_one, как в проде) -- ожидаемо чуть
    менее точен из-за bf16 rounding, это не баг кернеля, а особенность формата.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import mlx.core as mx

from rwkv_quant.formats.schema import QuantizedTensor
from rwkv_quant.formats.reader import _dequantize_one
from rwkv_quant.formats.writer import _real_quantize, _real_quantize_sparse_outlier
from rwkv_quant.backends.metal.quant_linear import QuantLinear

torch.manual_seed(0)
np.random.seed(0)


def _dequantize_fp32(qt) -> torch.Tensor:
    w = qt.codes.float() * qt.scale.float()
    if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
        rows, cols = qt.outlier_indices[:, 0].long(), qt.outlier_indices[:, 1].long()
        w[rows, cols] = qt.outlier_values.float()
    return w


def _run_case(name, out_features, in_features, bits, use_spqr, outlier_frac=0.02, n_batch=4):
    w = torch.randn(out_features, in_features) * torch.exp(torch.randn(out_features, 1) * 1.5)
    for r in range(0, out_features, 7):
        c = np.random.randint(0, in_features)
        w[r, c] *= 60.0

    if use_spqr:
        codes, scale, oi, ov = _real_quantize_sparse_outlier(w, bits, outlier_frac)
        qt = QuantizedTensor(key="test", group="proj", bits=bits, shape=tuple(w.shape),
                              codes=codes, scale=scale, outlier_indices=oi, outlier_values=ov)
    else:
        codes, scale = _real_quantize(w, bits)
        qt = QuantizedTensor(key="test", group="proj", bits=bits, shape=tuple(w.shape),
                              codes=codes, scale=scale)

    w_fp32 = _dequantize_fp32(qt)
    w_bf16 = _dequantize_one(qt).float()

    x = torch.randn(n_batch, in_features)
    y_fp32_ref = (x @ w_fp32.T).numpy()
    y_bf16_ref = (x @ w_bf16.T).numpy()

    ql = QuantLinear(qt)
    x_mx = mx.array(x.numpy())
    y_metal = np.array(ql(x_mx))

    def rel(a, b):
        return np.abs(a - b).max() / (np.abs(a).max() + 1e-8)

    err_vs_fp32 = rel(y_fp32_ref, y_metal)
    err_vs_bf16 = rel(y_bf16_ref, y_metal)
    status = "OK" if err_vs_fp32 < 1e-4 else "FAIL"
    print(f"[{status}] {name}: bits={bits} spqr={use_spqr} shape=({out_features},{in_features}) "
          f"err_vs_fp32={err_vs_fp32:.2e} err_vs_bf16(expected, informational)={err_vs_bf16:.2e}")
    return status == "OK"


def main():
    results = []
    results.append(_run_case("small_int8_plain",   out_features=64,  in_features=64,  bits=8, use_spqr=False))
    results.append(_run_case("small_int4_plain",   out_features=64,  in_features=64,  bits=4, use_spqr=False))
    results.append(_run_case("proj_int8_spqr",     out_features=512, in_features=512, bits=8, use_spqr=True, outlier_frac=0.02))
    results.append(_run_case("proj_int4_spqr",     out_features=512, in_features=512, bits=4, use_spqr=True, outlier_frac=0.02))
    results.append(_run_case("cmix_int4_spqr_big", out_features=768, in_features=768*4, bits=4, use_spqr=True, outlier_frac=0.01))
    results.append(_run_case("nonsquare_int8",     out_features=384, in_features=128,  bits=8, use_spqr=False))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
