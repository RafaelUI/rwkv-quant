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

from ...formats.schema import int8_codes
from .quant_linear import _build_outlier_csr

_kernel_cache = {}

TG = 32  # потоков на выходную фичу; 32 = один simdgroup, редукция без барьеров

GEMM_MIN_BATCH = 16  # N >= порога: dequant fp16 + mx.matmul (GEMM-путь префилла).
                     # GEMV-кернели перечитывают веса на КАЖДУЮ строку батча
                     # (нулевой реюз, см. NEXT_SESSION "Резервы"); dequant-путь
                     # платит ~4.5 байта/элемент однократно (read codes + write
                     # fp16 + matmul read) против N*(0.5-1) байт у GEMV --
                     # брейк-ивен ~N=8-16, дальше выигрыш растёт с N.


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


_packed_cache = {}

R_PACKED = 8  # выходных строк на threadgroup: x-активации читаются один раз
              # на R строк -- при GEMV именно суммарный трафик x (x читают ВСЕ
              # threadgroup'ы, OUT/R раз) упирается в L2, а не codes в DRAM.
              # Без блокировки packed был МЕДЛЕННЕЕ int8 (0.4-0.6x) при
              # вдвое меньших codes; эмпирика M4: R=8/TG=32 head 2x vs int8.


def _get_kernel_packed(IN: int, OUT: int, has_outliers: bool):
    """GEMV по biased split-нибблам (schema.pack_int4): threadgroup из TG
    потоков обслуживает R_PACKED строк, uchar4 = 8 колонок, распаковка --
    два векторных &0xF / >>4 без знакового расширения, поправка -8*sum(x)
    после цикла. Требования: IN % 8 == 0; OUT произвольный (guard)."""
    key = (IN, OUT, has_outliers)
    if key in _packed_cache:
        return _packed_cache[key]

    assert IN % 8 == 0, "packed-кернель требует IN % 8 == 0 (uchar4 = 8 колонок)"

    # guard на хвостовой неполный блок строк -- ветка в горячем цикле ломает
    # анроллинг (эмпирика: cmix 0.15 -> 0.31мс), поэтому компилируем её
    # только когда OUT не делится на R (наши шапки 1.5B все делятся).
    guard_hot  = "" if OUT % R_PACKED == 0 else "            if (row0 + j >= OUT_C) break;\n"
    guard_tail = "" if OUT % R_PACKED == 0 else "        if (row >= OUT_C) break;\n"

    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint TG    = {TG};
constant uint R     = {R_PACKED};
"""
    outlier_body = """
        uint start = row_offsets[row];
        uint end   = row_offsets[row+1];
        float oacc = 0.0f;
        for (uint idx = start + lane; idx < end; idx += TG) {
            uint c = (uint)outlier_cols[idx];
            oacc += x[n*IN_C + c] * outlier_vals[idx];
        }
        a += oacc;
""" if has_outliers else ""

    body = """
    uint g    = threadgroup_position_in_grid.x;   // блок из R строк
    uint n    = threadgroup_position_in_grid.y;   // batch row
    uint lane = thread_position_in_threadgroup.x;
    uint row0 = g * R;

    device const float4* x4 = (device const float4*)(x + n*IN_C);
    float acc[R];
    for (uint j = 0; j < R; j++) acc[j] = 0.0f;
    // sum(x) для biased-поправки предвычислен снаружи (xsum[n]): раньше
    // каждый из OUT/R threadgroup'ов считал его заново -- ~11% ALU кернеля
    // (2 dot из 18 на итерацию) на одно и то же число.
    float xs = xsum[n] / (float)TG;  // делим на TG: каждый lane вычтет свою долю до simd_sum

    for (uint p = lane; p < IN_C/8; p += TG) {
        float4 xa = x4[p], xb = x4[IN_C/8 + p];
        for (uint j = 0; j < R; j++) {
GUARD_HOT            uchar4 q = ((device const uchar4*)(codes + (row0+j)*(IN_C/2)))[p];
            uchar4 lo = q & (uchar)0xF;
            uchar4 hi = q >> 4;
            acc[j] += dot(xa, float4(lo.x, lo.y, lo.z, lo.w))
                    + dot(xb, float4(hi.x, hi.y, hi.z, hi.w));
        }
    }
    for (uint j = 0; j < R; j++) {
        uint row = row0 + j;
GUARD_TAIL        // biased: sum(x*(n-8)) = sum(x*n) - 8*sum(x); частичные суммы по
        // lane'ам согласованы (каждый lane вычитает свои 8*xs), scale --
        // per-row константа, домножение до simd_sum эквивалентно.
        float a = (acc[j] - 8.0f * xs) * (float)scale[row];
""" + outlier_body + """
        a = simd_sum(a);
        if (lane == 0)
            out[n*OUT_C + row] = a;
    }
"""
    body = body.replace("GUARD_HOT", guard_hot).replace("GUARD_TAIL", guard_tail)
    kern = mx.fast.metal_kernel(
        name=f"quant_linear_v2p_{'spqr' if has_outliers else 'plain'}_{IN}_{OUT}",
        input_names=["x", "codes", "scale", "row_offsets", "outlier_cols", "outlier_vals", "xsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _packed_cache[key] = kern
    return kern


class QuantLinearV2:
    """Drop-in замена QuantLinear (v1) с threadgroup-редукцией."""

    def __init__(self, qt):
        assert qt.bits < 16, "QuantLinearV2 только для квантованных тензоров"
        self.out_features, self.in_features = qt.shape
        self.packed = qt.codes_packed is not None and self.in_features % 8 == 0
        if self.packed:
            self.codes = mx.array(qt.codes_packed.numpy())  # uint8 нибблы, половина трафика
        elif qt.codes is not None:
            self.codes = mx.array(qt.codes.numpy())
        else:  # packed-тензор с IN, не кратным 8 -- fallback на распаковку
            self.codes = mx.array(int8_codes(qt).numpy())
        self.scale = mx.array(qt.scale.float().numpy().reshape(-1))
        self.row_offsets, self.outlier_cols, self.outlier_vals = _build_outlier_csr(
            qt.outlier_indices, qt.outlier_values, self.out_features)
        self.has_outliers = qt.outlier_indices is not None and qt.outlier_indices.numel() > 0
        if self.has_outliers:
            oidx = qt.outlier_indices.numpy()
            self._out_rows = mx.array(oidx[:, 0].astype(np.int32))
            self._out_cols = mx.array(oidx[:, 1].astype(np.int32))
            self._out_vals = mx.array(qt.outlier_values.float().numpy())

    def _dequant_w(self):
        """codes(+scale+outliers) -> fp16 [out, in] на GPU. Временный тензор
        на один вызов (не кешируется: резидентный fp16 съел бы весь выигрыш
        по памяти от квантования). codes нулевые в outlier-позициях
        (writer._real_quantize_sparse_outlier), поэтому scatter-add --
        чистое дополнение. ВСЯ арифметика в fp16: int-коды <= 127 в fp16
        точны, потеря только на округлении scale/произведения (та же, что
        у fp16-dense пути); fp32-промежутки здесь стоили бы 512MB
        транзиента на head (65536x2048) на каждый вызов."""
        if self.packed:
            lo = (self.codes & 0xF).astype(mx.float16) - 8.0
            hi = (self.codes >> 4).astype(mx.float16) - 8.0
            w = mx.concatenate([lo, hi], axis=1)[:, : self.in_features]
        else:
            w = self.codes.astype(mx.float16)
        w = w * self.scale.astype(mx.float16)[:, None]
        if self.has_outliers:
            w = w.at[self._out_rows, self._out_cols].add(self._out_vals.astype(mx.float16))
        return w

    def __call__(self, x):
        lead_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).astype(mx.float32)
        N = x2d.shape[0]
        if N >= GEMM_MIN_BATCH:
            # GEMM-путь префилла: один dequant + оптимизированный matmul MLX
            # с реюзом весов по батчу; fp16-операнды (как dense-путь модели),
            # выход fp32 -- интерфейс идентичен GEMV-веткам.
            w = self._dequant_w()
            out = mx.matmul(x2d.astype(mx.float16), w.T).astype(mx.float32)
            return out.reshape(*lead_shape, self.out_features)
        getk = _get_kernel_packed if self.packed else _get_kernel_v2
        kern = getk(self.in_features, self.out_features, self.has_outliers)
        if self.packed:
            n_groups = (self.out_features + R_PACKED - 1) // R_PACKED
            extra = [mx.sum(x2d, axis=-1)]  # xsum[n]: один мелкий кернель вместо OUT/R пересчётов внутри GEMV
        else:
            n_groups = self.out_features
            extra = []
        out = kern(
            inputs=[x2d, self.codes, self.scale, self.row_offsets, self.outlier_cols, self.outlier_vals] + extra,
            grid=(n_groups * TG, N, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(N, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]
        return out.reshape(*lead_shape, self.out_features)
