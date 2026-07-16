"""
quant_linear_v2.py — оптимизированный GEMV для int8 codes + per-row scale
(+ SpQR), threadgroup-параллельная редукция вместо serial per-thread цикла v1.

Отличия от v1 (quant_linear.py):
  - один threadgroup из TG потоков на выходную фичу (v1: 1 поток на фичу);
  - потоки шагают по входу со страйдом TG, codes читаются векторно (char4,
    4 int8 за одну загрузку) и коалесцированно между соседними потоками;
  - редукция частичных сумм через simd_sum (+ threadgroup-память между
    simdgroup'ами, если TG > 32);
  - SpQR-хвост обрабатывается тем же threadgroup'ом (страйд по CSR-диапазону
    строки) и входит в ту же редукцию.

Интерфейс идентичен v1 QuantLinear — численную эквивалентность проверяет
tests/test_quant_linear_v2.py.
"""
import numpy as np
import mlx.core as mx

from .quant_linear import _build_outlier_csr

_kernel_cache = {}

TG = 32  # потоков на выходную фичу; 32 = один simdgroup, редукция без барьеров


def _get_kernel_v2(IN: int, OUT: int, has_outliers: bool):
    key = (IN, OUT, has_outliers)
    if key in _kernel_cache:
        return _kernel_cache[key]

    assert IN % 4 == 0, "IN должен делиться на 4 (char4-загрузки)"

    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint TG    = {TG};
"""
    outlier_body = """
    // SpQR: тот же threadgroup шагает по outlier'ам строки
    uint start = row_offsets[oi];
    uint end   = row_offsets[oi+1];
    for (uint idx = start + lane; idx < end; idx += TG) {
        uint c = (uint)outlier_cols[idx];
        acc += x[n*IN_C + c] * outlier_vals[idx];
    }
""" if has_outliers else ""

    body = """
    uint oi   = threadgroup_position_in_grid.x;   // выходная фича
    uint n    = threadgroup_position_in_grid.y;   // batch row
    uint lane = thread_position_in_threadgroup.x; // 0..TG-1 (TG==32: один simdgroup)

    device const char4* codes4 = (device const char4*)(codes + oi*IN_C);
    device const float4* x4    = (device const float4*)(x + n*IN_C);

    float acc = 0.0f;
    for (uint c = lane; c < IN_C/4; c += TG) {
        char4  q  = codes4[c];
        float4 xv = x4[c];
        acc += xv.x*(float)q.x + xv.y*(float)q.y + xv.z*(float)q.z + xv.w*(float)q.w;
    }
    acc *= (float)scale[oi];
""" + outlier_body + """
    acc = simd_sum(acc);
    if (lane == 0)
        out[n*OUT_C + oi] = acc;
"""
    kern = mx.fast.metal_kernel(
        name=f"quant_linear_v2_{'spqr' if has_outliers else 'plain'}_{IN}_{OUT}",
        input_names=["x", "codes", "scale", "row_offsets", "outlier_cols", "outlier_vals"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _kernel_cache[key] = kern
    return kern


class QuantLinearV2:
    """Drop-in замена QuantLinear (v1) с threadgroup-редукцией."""

    def __init__(self, qt):
        assert qt.bits < 16, "QuantLinearV2 только для квантованных тензоров"
        self.out_features, self.in_features = qt.shape
        self.codes = mx.array(qt.codes.numpy())
        self.scale = mx.array(qt.scale.float().numpy().reshape(-1))
        self.row_offsets, self.outlier_cols, self.outlier_vals = _build_outlier_csr(
            qt.outlier_indices, qt.outlier_values, self.out_features)
        self.has_outliers = qt.outlier_indices is not None and qt.outlier_indices.numel() > 0

    def __call__(self, x):
        lead_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).astype(mx.float32)
        N = x2d.shape[0]
        kern = _get_kernel_v2(self.in_features, self.out_features, self.has_outliers)
        out = kern(
            inputs=[x2d, self.codes, self.scale, self.row_offsets, self.outlier_cols, self.outlier_vals],
            grid=(self.out_features * TG, N, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(N, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]
        return out.reshape(*lead_shape, self.out_features)
