"""Кернель-3 фаза C: интерлив-репак при загрузке (дисковый формат нетронут).
Раскладка на блок (row, p): codes 16B [+ qh 4B [+ qh2 4B]] контигуозно
(16/20/24B); qs+qm -> uchar2[OUT*NB]; d+dm -> half2[OUT*NSB].
Транзакций на (row, block): 7 -> 4-5. Поверх раскладки фазы B
(NSG simdgroups x RS строк). Порядок математики прежний.
A/B в одном процессе; печатается maxreldiff vs старый кернель.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.backends.metal.quant_linear_gw import _get_kernel_gw, TG, R

_cache = {}


def get_kernel_k3c(IN, OUT, xbits, NSG=2, RS=2):
    key = (IN, OUT, xbits, NSG, RS)
    if key in _cache:
        return _cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    NB, NSB = IN // 32, IN // 256
    SU = 4 + xbits  # стрйд блока в uint'ах: 16/20/24Б
    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
constant uint NSG   = {NSG};
constant uint RS    = {RS};
constant uint SU    = {SU};
"""
    hb_decode = """
            uint hb = qb[4];
            l0 |= as_type<uchar4>((((hb >>  0) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            l1 |= as_type<uchar4>((((hb >>  4) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            l2 |= as_type<uchar4>((((hb >>  8) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            l3 |= as_type<uchar4>((((hb >> 12) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            h0 |= as_type<uchar4>((((hb >> 16) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            h1 |= as_type<uchar4>((((hb >> 20) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            h2 |= as_type<uchar4>((((hb >> 24) & 0xFu) * 0x00204081u & 0x01010101u) << 4);
            h3 |= as_type<uchar4>((( hb >> 28)         * 0x00204081u & 0x01010101u) << 4);
""" if xbits >= 1 else ""
    hb2_decode = """
            uint hb2 = qb[5];
            l0 |= as_type<uchar4>((((hb2 >>  0) & 0xFu) * 0x00204081u & 0x01010101u) << 5);
            l1 |= as_type<uchar4>((((hb2 >>  4) & 0xFu) * 0x00204081u & 0x01010101u) << 5);
            l2 |= as_type<uchar4>((((hb2 >>  8) & 0xFu) * 0x00204081u & 0x01010101u) << 5);
            l3 |= as_type<uchar4>((((hb2 >> 12) & 0xFu) * 0x00204081u & 0x01010101u) << 5);
            h0 |= as_type<uchar4>((((hb2 >> 16) & 0xFu) * 0x00204081u & 0x01010101u) << 5);
            h1 |= as_type<uchar4>((((hb2 >> 20) & 0xFu) * 0x00204081u & 0x01010101u) << 5);
            h2 |= as_type<uchar4>((((hb2 >> 24) & 0xFu) * 0x00204081u & 0x01010101u) << 5);
            h3 |= as_type<uchar4>((( hb2 >> 28)         * 0x00204081u & 0x01010101u) << 5);
""" if xbits >= 2 else ""
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint row0 = tgid * (NSG * RS) + sg * RS;

    device const float4* x4 = (device const float4*)x;
    device const uint*   qu = (device const uint*)qblk;
    device const uchar2* sm2 = (device const uchar2*)qsqm;
    device const half2*  dd2 = (device const half2*)ddm;
    float acc[RS];
    for (uint j = 0; j < RS; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += 32) {
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float xbs = xbsum[p];
        for (uint j = 0; j < RS; j++) {
            device const uint* qb = qu + ((row0+j)*NB + p) * SU;
            uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
""" + hb_decode + hb2_decode + """
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
        name=f"k3d_gw{4 + xbits}_s{NSG}r{RS}_{IN}_{OUT}",
        input_names=["x", "qblk", "qsqm", "ddm", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _cache[key] = kern
    return kern


class Syn:
    def __init__(self, OUT, IN, xbits, rng):
        self.OUT, self.IN, self.xbits = OUT, IN, xbits
        NB, NSB = IN // 32, IN // 256
        self.NB, self.NSB = NB, NSB
        codes = rng.integers(0, 256, (OUT, IN // 2), dtype=np.uint8)
        qs = rng.integers(0, 64, (OUT, NB), dtype=np.uint8)
        qm = rng.integers(-31, 32, (OUT, NB)).astype(np.int8)
        d = (rng.random((OUT, NSB), dtype=np.float32) * 1e-3 + 1e-3).astype(np.float16)
        dm = (rng.random((OUT, NSB), dtype=np.float32) * 1e-3 + 1e-3).astype(np.float16)
        qh = rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8)
        qh2 = rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8)
        self.codes = mx.array(codes)
        self.qs, self.qm = mx.array(qs), mx.array(qm)
        self.d, self.dm = mx.array(d), mx.array(dm)
        self.qh = mx.array(qh) if xbits >= 1 else mx.zeros((1,), dtype=mx.uint8)
        self.qh2 = mx.array(qh2) if xbits >= 2 else mx.zeros((1,), dtype=mx.uint8)
        # интерлив
        parts = [codes.reshape(OUT, NB, 16)]
        if xbits >= 1: parts.append(qh.reshape(OUT, NB, 4))
        if xbits >= 2: parts.append(qh2.reshape(OUT, NB, 4))
        self.qblk = mx.array(np.ascontiguousarray(
            np.concatenate(parts, axis=2).reshape(OUT, -1)))
        self.qsqm = mx.array(np.ascontiguousarray(
            np.stack([qs, qm.view(np.uint8)], axis=-1).reshape(OUT, -1)))
        self.ddm = mx.array(np.ascontiguousarray(
            np.stack([d, dm], axis=-1).reshape(OUT, -1)))
        mx.eval(self.codes, self.qs, self.qm, self.d, self.dm, self.qh,
                self.qh2, self.qblk, self.qsqm, self.ddm)
        self.mb = (codes.size + qs.size + qm.size + d.size*2 + dm.size*2
                   + (qh.size if xbits >= 1 else 0)
                   + (qh2.size if xbits >= 2 else 0)) / 1e6

    def old(self, x, xbsum):
        kern = _get_kernel_gw(self.IN, self.OUT, self.xbits)
        n_groups = (self.OUT + R - 1) // R
        return kern(
            inputs=[x, self.codes, self.qs, self.qm, self.d, self.dm,
                    self.qh, self.qh2, xbsum],
            grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(1, self.OUT)], output_dtypes=[mx.float32])[0]

    def k3c(self, x, xbsum, NSG, RS):
        kern = get_kernel_k3c(self.IN, self.OUT, self.xbits, NSG, RS)
        n_tg = self.OUT // (NSG * RS)
        return kern(
            inputs=[x, self.qblk, self.qsqm, self.ddm, xbsum],
            grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
            output_shapes=[(1, self.OUT)], output_dtypes=[mx.float32])[0]


def bench(fn, reps=8, warm=4, calls=8):
    for _ in range(warm):
        mx.eval(*[fn() for _ in range(calls)])
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn() for _ in range(calls)])
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3 / calls)
    return ts


CASES = [
    ("tmix5 2048x2048",  2048, 2048, 1),
    ("tmix6 2048x2048",  2048, 2048, 2),
    ("cmixK4 8192x2048", 8192, 2048, 0),
    ("cmixK6 8192x2048", 8192, 2048, 2),
    ("cmixV4 2048x8192", 2048, 8192, 0),
    ("cmixV6 2048x8192", 2048, 8192, 2),
    ("head5 65536x2048", 65536, 2048, 1),
    ("head6 65536x2048", 65536, 2048, 2),
]
VARIANTS = [(2, 2), (2, 4), (2, 8), (4, 4)]

rng = np.random.default_rng(0)
print("=== кернель-3 фаза D: интерлив + мульт-трюк битплоскостей ===")
for name, OUT, IN, xbits in CASES:
    t = Syn(OUT, IN, xbits, rng)
    x = mx.array(rng.standard_normal((1, IN)).astype(np.float32))
    xbsum = mx.sum(x.reshape(1, t.NB, 32), axis=2)
    mx.eval(x, xbsum)

    ref = np.array(t.old(x, xbsum))
    denom = np.maximum(np.abs(ref), 1e-6)
    diffs = {}
    for NSG, RS in VARIANTS:
        if OUT % (NSG * RS):
            continue
        got = np.array(t.k3c(x, xbsum, NSG, RS))
        diffs[(NSG, RS)] = float(np.max(np.abs(got - ref) / denom))

    res = {"old": []}
    for v in diffs:
        res[v] = []
    for _ in range(3):
        res["old"] += bench(lambda: t.old(x, xbsum))
        for v in diffs:
            res[v] += bench(lambda: t.k3c(x, xbsum, *v))
    mo = float(np.median(res["old"]))
    line = f"{name:18s} old={mo:6.3f}ms [{t.mb/mo:5.1f}GB/s]"
    for v in diffs:
        m = float(np.median(res[v]))
        tag = "BIT=" if diffs[v] == 0.0 else f"rd{diffs[v]:.0e}"
        line += f"  s{v[0]}r{v[1]}={m:6.3f} [{t.mb/m:5.1f}] x{mo/m:4.2f} {tag}"
    print(line, flush=True)
    del t
