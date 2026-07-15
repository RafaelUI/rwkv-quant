"""
backends/metal/quant_linear.py — INT8 per-row scale + SpQR sparse-outlier
Linear на Metal (MLX custom kernel), без промежуточного bf16-разворачивания.

Заменяет formats.reader.load_dequantized() (весь тензор int8 -> bf16 в
памяти) на прямой матмул по codes/scale + точечную поправку по
разреженным outlier'ам. codes уже нулевые в outlier-позициях (см.
writer._real_quantize_sparse_outlier), поэтому outlier-коррекция — это
чистое дополнение, а не overwrite.
"""
import numpy as np
import mlx.core as mx


def _build_outlier_csr(outlier_indices, outlier_values, n_rows):
    """torch [n_outliers,2] (row,col) + [n_outliers] значения ->
    row_offsets[n_rows+1], cols[n_outliers], vals[n_outliers] — отсортировано
    по row, для CSR-обхода внутри kernel'я. Выполняется один раз при загрузке."""
    if outlier_indices is None or outlier_indices.numel() == 0:
        return (mx.array(np.zeros(n_rows + 1, dtype=np.int32)),
                mx.array(np.zeros(1, dtype=np.int32)),
                mx.array(np.zeros(1, dtype=np.float32)))

    rows = outlier_indices[:, 0].numpy()
    cols = outlier_indices[:, 1].numpy()
    vals = outlier_values.float().numpy()

    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]

    row_offsets = np.zeros(n_rows + 1, dtype=np.int32)
    row_offsets[1:] = np.cumsum(np.bincount(rows, minlength=n_rows))

    return (mx.array(row_offsets), mx.array(cols.astype(np.int32)), mx.array(vals.astype(np.float32)))


_kernel_cache = {}


def _get_kernel(IN: int, OUT: int, has_outliers: bool):
    key = (IN, OUT, has_outliers)
    if key in _kernel_cache:
        return _kernel_cache[key]

    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
"""
    outlier_body = """
    uint start = row_offsets[oi];
    uint end   = row_offsets[oi+1];
    for (uint idx = start; idx < end; idx++) {
        uint c = (uint)outlier_cols[idx];
        acc += x[n*IN_C + c] * outlier_vals[idx];
    }
""" if has_outliers else ""

    body = """
    uint oi = thread_position_in_grid.x;   // output feature
    uint n  = thread_position_in_grid.y;   // batch row

    float acc = 0.0f;
    for (uint c = 0; c < IN_C; c++)
        acc += x[n*IN_C + c] * (float)codes[oi*IN_C + c];
    acc *= (float)scale[oi];
""" + outlier_body + """
    out[n*OUT_C + oi] = acc;
"""
    kern = mx.fast.metal_kernel(
        name=f"quant_linear_{'spqr' if has_outliers else 'plain'}_{IN}_{OUT}",
        input_names=["x", "codes", "scale", "row_offsets", "outlier_cols", "outlier_vals"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _kernel_cache[key] = kern
    return kern


class QuantLinear:
    """Один Linear на int8 codes + per-row scale (+ опц. SpQR), веса живут
    на GPU в упакованном виде — никогда не разворачиваются в полный bf16."""

    def __init__(self, qt):
        # qt: rwkv_quant.formats.schema.QuantizedTensor, bits < 16
        assert qt.bits < 16, "QuantLinear только для квантованных тензоров"
        self.out_features, self.in_features = qt.shape
        self.codes = mx.array(qt.codes.numpy())          # int8 [out, in]
        self.scale = mx.array(qt.scale.float().numpy().reshape(-1))  # fp32 [out]
        self.row_offsets, self.outlier_cols, self.outlier_vals = _build_outlier_csr(
            qt.outlier_indices, qt.outlier_values, self.out_features)
        self.has_outliers = qt.outlier_indices is not None and qt.outlier_indices.numel() > 0

    def __call__(self, x):
        lead_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).astype(mx.float32)
        N = x2d.shape[0]
        kern = _get_kernel(self.in_features, self.out_features, self.has_outliers)
        out = kern(
            inputs=[x2d, self.codes, self.scale, self.row_offsets, self.outlier_cols, self.outlier_vals],
            grid=(self.out_features, N, 1), threadgroup=(1, 1, 1),
            output_shapes=[(N, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]
        return out.reshape(*lead_shape, self.out_features)
