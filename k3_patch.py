"""Патч quant_linear_gw.py: кернель-3 (раскладка MLX qmv поверх sb6).
Старые билдеры не трогаем; добавляем новые + переписываем классы.
Применение: venv/bin/python k3_patch.py (из tests/ или корня)."""
import re, sys, os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "rwkv_quant/backends/metal/quant_linear_gw.py")
if not os.path.exists(PATH):
    PATH = os.path.expanduser(
        "~/Develop/rwkv-quant/rwkv_quant/backends/metal/quant_linear_gw.py")
src = open(PATH).read()
assert "K3 =" not in src, "патч уже применён"

K3_BLOCK = '''

# ---------------------------------------------------------------------------
# КЕРНЕЛЬ-3 (19.07): раскладка MLX qmv (quantized.h, PR #1503) поверх
# РОДНОГО sb6-формата (дисковый формат нетронут, репак при загрузке):
# 1) NSG simdgroups x RS строк на threadgroup (вместо 1 simd x R=8):
#    вдвое меньше регистров/ALU на поток, лучше сокрытие латентности;
# 2) интерлив: qblk = codes 16Б [+qh 4Б [+qh2 4Б]] контигуозно на блок,
#    qsqm = uchar2 (qs,qm) на блок, ddm = half2 (d,dm) на суперблок:
#    4-5 транзакций на (строку, блок) вместо 7 из семи разных потоков;
# 3) мульт-трюк битплоскостей: ниббл -> 4 байта одним умножением
#    ((nib * 0x00204081) & 0x01010101), ~3x меньше ALU на плоскость.
# Порядок математики на lane идентичен старому кернелю => БИТ-В-БИТ
# (проверено по всем формам/вариантам, bench_kernel3_proto{,C,D,E}.py).
# Свипы 19.07: N=1 tmix x1.21-1.25, cmixK x1.13-1.17, cmixV x1.07-1.14,
# head x1.05-1.17 (101-103 GB/s); NB=4 tmix x1.19, cmixK6 x1.34,
# head x1.40. Память 1x: старые буферы не хранятся (ленивые view
# через __getattr__ для _dequant_w/бенчей).

K3 = True


def _k3_cfg(IN, OUT, xbits):
    """(NSG, RS) для N=1 GEMV -- свип bench_kernel3_protoD.py."""
    if OUT >= 32768:
        return (4, 4)                              # head
    if OUT >= 8192:
        return (2, 2) if xbits == 0 else (4, 4)    # cmix key
    return (4, 4)                                  # tmix / cmix value


def _k3_cfg_nb(IN, OUT, xbits):
    """(NSG, RS) для N-батча (verify) -- свип bench_kernel3_protoE.py."""
    if OUT >= 32768:
        return (4, 2)
    if OUT >= 8192:
        return (2, 4) if xbits == 0 else (4, 2)
    if IN >= 8192:
        return (2, 4)
    return (4, 4)


def _k3_plane(src, shift):
    regs = ["l0", "l1", "l2", "l3", "h0", "h1", "h2", "h3"]
    ls = []
    for i, reg in enumerate(regs):
        sh = i * 4
        if sh == 0:
            nib = f"({src} & 0xFu)"
        elif sh == 28:
            nib = f"({src} >> 28)"
        else:
            nib = f"(({src} >> {sh}) & 0xFu)"
        ls.append(f"            {reg} |= as_type<uchar4>(({nib} * 0x00204081u"
                  f" & 0x01010101u) << {shift});")
    return "\\n".join(ls) + "\\n"


_K3_DECODE = """
            uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
"""


def _k3_hdr(IN, OUT, xbits, NSG, RS, extra=""):
    NB, NSB = IN // 32, IN // 256
    return f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
constant uint NSG   = {NSG};
constant uint RS    = {RS};
constant uint SU    = {4 + xbits};
{extra}"""


def _get_kernel_k3(IN, OUT, xbits, NSG, RS, out_per=0):
    """N=1 GEMV (out_per=0) либо фьюз r/k/v (out_per>0, вход-стек)."""
    assert xbits in (0, 1, 2)
    key = ("k3", IN, OUT, xbits, NSG, RS, out_per)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    op = out_per if out_per else OUT
    assert op % (NSG * RS) == 0
    hdr = _k3_hdr(IN, OUT, xbits, NSG, RS, f"constant uint OUT_PER = {op};")
    dec = _K3_DECODE
    if xbits >= 1:
        dec += "            uint hb = qb[4];\\n" + _k3_plane("hb", 4)
    if xbits >= 2:
        dec += "            uint hb2 = qb[5];\\n" + _k3_plane("hb2", 5)
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint row0 = tgid * (NSG * RS) + sg * RS;
    uint xi   = row0 / OUT_PER;

    device const float4* x4  = (device const float4*)(x + xi*IN_C);
    device const uint*   qu  = (device const uint*)qblk;
    device const uchar2* sm2 = (device const uchar2*)qsqm;
    device const half2*  dd2 = (device const half2*)ddm;
    float acc[RS];
    for (uint j = 0; j < RS; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += 32) {
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float xbs = xbsum[xi*NB + p];
        for (uint j = 0; j < RS; j++) {
            device const uint* qb = qu + ((row0+j)*NB + p) * SU;
""" + dec + """
            float dv = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                     + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                     + dot(xa2, float4(l2.x, l2.y, l2.z, l2.w))
                     + dot(xa3, float4(l3.x, l3.y, l3.z, l3.w))
                     + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                     + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w))
                     + dot(xb2, float4(h2.x, h2.y, h2.z, h2.w))
                     + dot(xb3, float4(h3.x, h3.y, h3.z, h3.w));
            uchar2 sm = sm2[(row0+j)*NB + p];
            half2  dd = dd2[(row0+j)*NSB + p/8];
            half  s  = (half)((float)sm.x * (float)dd.x);
            half  mn = (half)((float)as_type<char>(sm.y) * (float)dd.y);
            acc[j] += (float)s * dv + (float)mn * xbs;
        }
    }
    for (uint j = 0; j < RS; j++) {
        float a = simd_sum(acc[j]);
        if (lane == 0)
            out[row0 + j] = a;
    }
"""
    kern = mx.fast.metal_kernel(
        name=f"k3_gw{4 + xbits}_s{NSG}r{RS}_{IN}_{OUT}_{op}",
        input_names=["x", "qblk", "qsqm", "ddm", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern


def _get_kernel_k3nb(IN, OUT, xbits, NN, NSG, RS):
    """N-батчевый GEMV кернеля-3: веса блока декодируются один раз на NN
    колонок (структура _get_kernel_gw_nb, раскладка кернеля-3)."""
    assert xbits in (0, 1, 2) and NN >= 2
    key = ("k3nb", IN, OUT, xbits, NN, NSG, RS)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    hdr = _k3_hdr(IN, OUT, xbits, NSG, RS, f"constant uint NN = {NN};")
    dec = _K3_DECODE
    if xbits >= 1:
        dec += "            uint hb = qb[4];\\n" + _k3_plane("hb", 4)
    if xbits >= 2:
        dec += "            uint hb2 = qb[5];\\n" + _k3_plane("hb2", 5)
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint row0 = tgid * (NSG * RS) + sg * RS;

    device const uint*   qu  = (device const uint*)qblk;
    device const uchar2* sm2 = (device const uchar2*)qsqm;
    device const half2*  dd2 = (device const half2*)ddm;
    float acc[RS * NN];
    for (uint j = 0; j < RS * NN; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += 32) {
        for (uint j = 0; j < RS; j++) {
            device const uint* qb = qu + ((row0+j)*NB + p) * SU;
""" + dec + """
            float4 w0 = float4(l0.x, l0.y, l0.z, l0.w);
            float4 w1 = float4(l1.x, l1.y, l1.z, l1.w);
            float4 w2 = float4(l2.x, l2.y, l2.z, l2.w);
            float4 w3 = float4(l3.x, l3.y, l3.z, l3.w);
            float4 w4 = float4(h0.x, h0.y, h0.z, h0.w);
            float4 w5 = float4(h1.x, h1.y, h1.z, h1.w);
            float4 w6 = float4(h2.x, h2.y, h2.z, h2.w);
            float4 w7 = float4(h3.x, h3.y, h3.z, h3.w);
            uchar2 sm = sm2[(row0+j)*NB + p];
            half2  dd = dd2[(row0+j)*NSB + p/8];
            half  s  = (half)((float)sm.x * (float)dd.x);
            half  mn = (half)((float)as_type<char>(sm.y) * (float)dd.y);
            for (uint n = 0; n < NN; n++) {
                device const float4* x4 = (device const float4*)(x + n*IN_C);
                float dv = dot(x4[p*8+0], w0)
                         + dot(x4[p*8+1], w1)
                         + dot(x4[p*8+2], w2)
                         + dot(x4[p*8+3], w3)
                         + dot(x4[p*8+4], w4)
                         + dot(x4[p*8+5], w5)
                         + dot(x4[p*8+6], w6)
                         + dot(x4[p*8+7], w7);
                acc[j*NN + n] += (float)s * dv + (float)mn * xbsum[n*NB + p];
            }
        }
    }
    for (uint j = 0; j < RS; j++) {
        for (uint n = 0; n < NN; n++) {
            float a = simd_sum(acc[j*NN + n]);
            if (lane == 0)
                out[n*OUT_C + row0 + j] = a;
        }
    }
"""
    kern = mx.fast.metal_kernel(
        name=f"k3nb_gw{4 + xbits}_n{NN}s{NSG}r{RS}_{IN}_{OUT}",
        input_names=["x", "qblk", "qsqm", "ddm", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern

'''

NEW_CLASSES = '''class GwQuantLinear:
    """Linear по sb6-тензору формата v2 (bits 4/5/6). Интерфейс как у
    QuantLinearV2: __call__(x [..., IN]) -> [..., OUT] fp32.
    K3: хранит интерлив-буферы qblk/qsqm/ddm (память 1x); старые имена
    (codes/qs/qm/d/dm/qh/qh2) доступны как ленивые view (__getattr__)."""

    _COMPAT = ("codes", "qs", "qm", "d", "dm", "qh", "qh2")

    def __init__(self, qt):
        assert qt.gw_mode == "sb6"
        self.out_features, self.in_features = qt.shape
        OUT, IN = qt.shape
        self.NB, self.NSB = IN // 32, IN // 256
        self.xbits = (2 if qt.gw_qh2 is not None
                      else (1 if qt.gw_qh is not None else 0))
        self.has_qh = self.xbits >= 1  # обратная совместимость (Fused-ассерт)
        qs_np = unpack6(qt.gw_qsqm[..., :6], 8).reshape(OUT, self.NB).numpy()
        qm_np = (unpack6(qt.gw_qsqm[..., 6:], 8).reshape(OUT, self.NB)
                 .to(torch.int16) - 31).to(torch.int8).numpy()
        codes_np = qt.codes_packed.numpy()
        d_np, dm_np = qt.gw_d.numpy(), qt.gw_dm.numpy()
        qh_np = qt.gw_qh.numpy() if self.xbits >= 1 else None
        qh2_np = qt.gw_qh2.numpy() if self.xbits >= 2 else None
        self._k3 = bool(K3) and OUT % 16 == 0
        if self._k3:
            parts = [codes_np.reshape(OUT, self.NB, 16)]
            if self.xbits >= 1:
                parts.append(qh_np.reshape(OUT, self.NB, 4))
            if self.xbits >= 2:
                parts.append(qh2_np.reshape(OUT, self.NB, 4))
            self.qblk = mx.array(np.ascontiguousarray(
                np.concatenate(parts, axis=2).reshape(OUT, -1)))
            self.qsqm = mx.array(np.ascontiguousarray(
                np.stack([qs_np, qm_np.view(np.uint8)], axis=-1)
                .reshape(OUT, -1)))
            self.ddm = mx.array(np.ascontiguousarray(
                np.stack([d_np, dm_np], axis=-1).reshape(OUT, -1)))
            mx.eval(self.qblk, self.qsqm, self.ddm)
        else:
            self.codes = mx.array(codes_np)
            self.qs = mx.array(qs_np)
            self.qm = mx.array(qm_np)
            self.d = mx.array(d_np)
            self.dm = mx.array(dm_np)
            self.qh = (mx.array(qh_np) if self.xbits >= 1
                       else mx.zeros((1,), dtype=mx.uint8))
            self.qh2 = (mx.array(qh2_np) if self.xbits >= 2
                        else mx.zeros((1,), dtype=mx.uint8))

    def __getattr__(self, name):
        # ленивые view старых буферов из интерлива (только k3-режим)
        if name in GwQuantLinear._COMPAT and self.__dict__.get("_k3"):
            OUT, IN = self.out_features, self.in_features
            if name in ("codes", "qh", "qh2"):
                if name == "qh" and self.xbits < 1:
                    return mx.zeros((1,), dtype=mx.uint8)
                if name == "qh2" and self.xbits < 2:
                    return mx.zeros((1,), dtype=mx.uint8)
                blk = self.qblk.reshape(OUT, self.NB, 16 + 4 * self.xbits)
                if name == "codes":
                    return blk[:, :, :16].reshape(OUT, IN // 2)
                if name == "qh":
                    return blk[:, :, 16:20].reshape(OUT, IN // 8)
                return blk[:, :, 20:24].reshape(OUT, IN // 8)
            if name == "qs":
                return self.qsqm.reshape(OUT, self.NB, 2)[:, :, 0]
            if name == "qm":
                return mx.view(self.qsqm.reshape(OUT, self.NB, 2)[:, :, 1],
                               mx.int8)
            if name == "d":
                return self.ddm.reshape(OUT, self.NSB, 2)[:, :, 0]
            return self.ddm.reshape(OUT, self.NSB, 2)[:, :, 1]
        raise AttributeError(name)

    def _dequant_w(self):
        """sb6 -> fp16 [OUT, IN] на GPU для GEMM-префилла (транзиент на
        вызов, не кешируется -- см. примечание в QuantLinearV2)."""
        OUT, IN = self.out_features, self.in_features
        cb = self.codes.reshape(OUT, self.NB, 16)
        q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.float16)
        if self.xbits >= 1:
            bits = (self.qh[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
            bits = bits.reshape(OUT, IN).reshape(OUT, self.NB, 32)
            q = q + bits.astype(mx.float16) * 16.0
        if self.xbits >= 2:
            bits2 = (self.qh2[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
            bits2 = bits2.reshape(OUT, IN).reshape(OUT, self.NB, 32)
            q = q + bits2.astype(mx.float16) * 32.0
        s = (self.qs.astype(mx.float32).reshape(OUT, self.NSB, 8)
             * self.d.astype(mx.float32)[..., None]).astype(mx.float16)
        m = (self.qm.astype(mx.float32).reshape(OUT, self.NSB, 8)
             * self.dm.astype(mx.float32)[..., None]).astype(mx.float16)
        w = q * s.reshape(OUT, self.NB, 1) + m.reshape(OUT, self.NB, 1)
        return w.reshape(OUT, IN)

    def __call__(self, x):
        lead_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).astype(mx.float32)
        N = x2d.shape[0]
        if N >= GEMM_MIN_BATCH_NB:
            w = self._dequant_w()
            out = mx.matmul(x2d.astype(mx.float16), w.T).astype(mx.float32)
            return out.reshape(*lead_shape, self.out_features)
        if N > 1:
            outs = []
            i = 0
            while i < N:
                c = min(NB_CHUNK, N - i)
                if c == 1:
                    outs.append(self._gemv1(x2d[i:i+1]))
                else:
                    outs.append(self._gemv_nb(x2d[i:i+c], c))
                i += c
            out = mx.concatenate(outs, axis=0)
            return out.reshape(*lead_shape, self.out_features)
        return self._gemv1(x2d).reshape(*lead_shape, self.out_features)

    def _rb_nb(self, c):
        # свип 19.07 (bench_cmix_split): value-формы (IN>=8192) хотят rb8,
        # key-формы (OUT>=8192) rb4, остальное (tmix/head) rb2/таблица.
        if self.in_features >= 8192:
            return 8
        if self.out_features >= 8192:
            return 4
        return _RB_FOR_NN.get(c, RB_NB)

    def _gemv_nb(self, x2d, c):
        xbsum = mx.sum(x2d.reshape(c, self.NB, 32), axis=2)
        if self._k3 and not NB_V2:
            NSG, RS = _k3_cfg_nb(self.in_features, self.out_features,
                                 self.xbits)
            kern = _get_kernel_k3nb(self.in_features, self.out_features,
                                    self.xbits, c, NSG, RS)
            n_tg = self.out_features // (NSG * RS)
            return kern(
                inputs=[x2d, self.qblk, self.qsqm, self.ddm, xbsum],
                grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                output_shapes=[(c, self.out_features)],
                output_dtypes=[mx.float32],
            )[0]
        if NB_V2:
            rb = 2
            kern = _get_kernel_gw_nb2(self.in_features, self.out_features,
                                      self.xbits, c, rb)
        else:
            rb = self._rb_nb(c)
            kern = _get_kernel_gw_nb(self.in_features, self.out_features,
                                     self.xbits, c, rb)
        n_groups = (self.out_features + rb - 1) // rb
        return kern(
            inputs=[x2d, self.codes, self.qs, self.qm, self.d, self.dm,
                    self.qh, self.qh2, xbsum],
            grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(c, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]

    def _gemv1(self, x2d):
        xbsum = mx.sum(x2d.reshape(1, self.NB, 32), axis=2)
        if self._k3:
            NSG, RS = _k3_cfg(self.in_features, self.out_features, self.xbits)
            kern = _get_kernel_k3(self.in_features, self.out_features,
                                  self.xbits, NSG, RS)
            n_tg = self.out_features // (NSG * RS)
            return kern(
                inputs=[x2d, self.qblk, self.qsqm, self.ddm, xbsum],
                grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                output_shapes=[(1, self.out_features)],
                output_dtypes=[mx.float32],
            )[0]
        kern = _get_kernel_gw(self.in_features, self.out_features, self.xbits)
        n_groups = (self.out_features + R - 1) // R
        return kern(
            inputs=[x2d, self.codes, self.qs, self.qm, self.d, self.dm,
                    self.qh, self.qh2, xbsum],
            grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(1, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]


class GwQuantLinearFused:
    """Фьюз K однотипных GwQuantLinear (r/k/v proj) в один launch:
    конкатенация квантованных строк (формат нетронут), кернель выбирает
    вход по номеру строки. Только decode-путь (B*T=1 на вход), побитово
    идентичен K отдельным вызовам (та же математика строки).
    __call__(xstack [K, IN]) -> [K, out_per]."""

    def __init__(self, lins):
        l0 = lins[0]
        assert all(isinstance(l, GwQuantLinear) for l in lins)
        assert all(l.in_features == l0.in_features and
                   l.out_features == l0.out_features and
                   l.xbits == l0.xbits for l in lins)
        self.K = len(lins)
        self.out_per = l0.out_features
        self.out_features = self.out_per * self.K
        self.in_features = l0.in_features
        self.NB, self.NSB = l0.NB, l0.NSB
        self.xbits = l0.xbits
        self.has_qh = l0.has_qh
        self._k3 = all(getattr(l, "_k3", False) for l in lins)
        if self._k3:
            NSG, RS = _k3_cfg(self.in_features, self.out_per, self.xbits)
            self._k3 = self.out_per % (NSG * RS) == 0
        if self._k3:
            self.qblk = mx.concatenate([l.qblk for l in lins], axis=0)
            self.qsqm = mx.concatenate([l.qsqm for l in lins], axis=0)
            self.ddm = mx.concatenate([l.ddm for l in lins], axis=0)
            mx.eval(self.qblk, self.qsqm, self.ddm)
        else:
            self.codes = mx.concatenate([l.codes for l in lins], axis=0)
            self.qs = mx.concatenate([l.qs for l in lins], axis=0)
            self.qm = mx.concatenate([l.qm for l in lins], axis=0)
            self.d = mx.concatenate([l.d for l in lins], axis=0)
            self.dm = mx.concatenate([l.dm for l in lins], axis=0)
            self.qh = (mx.concatenate([l.qh for l in lins], axis=0)
                       if self.xbits >= 1 else mx.zeros((1,), dtype=mx.uint8))
            self.qh2 = (mx.concatenate([l.qh2 for l in lins], axis=0)
                        if self.xbits >= 2 else mx.zeros((1,), dtype=mx.uint8))

    def __call__(self, xstack):
        # xstack: [K, IN] fp32
        xbsum = mx.sum(xstack.reshape(self.K, self.NB, 32), axis=2)
        if self._k3:
            NSG, RS = _k3_cfg(self.in_features, self.out_per, self.xbits)
            kern = _get_kernel_k3(self.in_features, self.out_features,
                                  self.xbits, NSG, RS, out_per=self.out_per)
            n_tg = self.out_features // (NSG * RS)
            out = kern(
                inputs=[xstack, self.qblk, self.qsqm, self.ddm, xbsum],
                grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                output_shapes=[(1, self.out_features)],
                output_dtypes=[mx.float32],
            )[0]
            return out.reshape(self.K, self.out_per)
        kern = _get_kernel_gw(self.in_features, self.out_features,
                              self.xbits, out_per=self.out_per)
        n_groups = (self.out_features + R - 1) // R
        out = kern(
            inputs=[xstack, self.codes, self.qs, self.qm, self.d, self.dm,
                    self.qh, self.qh2, xbsum],
            grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(1, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]
        return out.reshape(self.K, self.out_per)
'''

# вырезаем старые классы (от "class GwQuantLinear:" до конца файла)
i = src.index("class GwQuantLinear:")
src_new = src[:i] + NEW_CLASSES
# вставляем кернель-3 перед классами
src_new = src_new.replace("class GwQuantLinear:", K3_BLOCK.rstrip() + "\n\n\nclass GwQuantLinear:", 1)
open(PATH, "w").write(src_new)
print("OK: патч применён,", len(src_new.splitlines()), "строк")
