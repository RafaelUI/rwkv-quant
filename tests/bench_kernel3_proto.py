"""Кернель-3 прототип, фаза B: раскладка MLX qmv_fast (NSG simdgroups/TG,
RS строк на simdgroup) поверх родного gw sb6-формата БЕЗ смены буферов.
Порядок математики на lane идентичен старому кернелю (p = lane + t*32,
те же 8 dot'ов, тот же simd_sum) => ожидание: бит-в-бит.
A/B-чередование в одном процессе (закон 1), синтетические тензоры.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.backends.metal.quant_linear_gw import _get_kernel_gw, TG, R

_cache = {}


def _qh_block(buf, shift):
    return f"""
            uint hb{shift} = ((device const uint*)({buf} + (row0+j)*(IN_C/8)))[p];
            l0 |= uchar4((uint4(hb{shift}) >> uint4( 0, 1, 2, 3)) & 1u) << {shift};
            l1 |= uchar4((uint4(hb{shift}) >> uint4( 4, 5, 6, 7)) & 1u) << {shift};
            l2 |= uchar4((uint4(hb{shift}) >> uint4( 8, 9,10,11)) & 1u) << {shift};
            l3 |= uchar4((uint4(hb{shift}) >> uint4(12,13,14,15)) & 1u) << {shift};
            h0 |= uchar4((uint4(hb{shift}) >> uint4(16,17,18,19)) & 1u) << {shift};
            h1 |= uchar4((uint4(hb{shift}) >> uint4(20,21,22,23)) & 1u) << {shift};
            h2 |= uchar4((uint4(hb{shift}) >> uint4(24,25,26,27)) & 1u) << {shift};
            h3 |= uchar4((uint4(hb{shift}) >> uint4(28,29,30,31)) & 1u) << {shift};
"""


def get_kernel_k3(IN, OUT, xbits, NSG=2, RS=4):
    key = (IN, OUT, xbits, NSG, RS)
    if key in _cache:
        return _cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    NB, NSB = IN // 32, IN // 256
    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
constant uint NSG   = {NSG};
constant uint RS    = {RS};
"""
    qh_body = _qh_block("qh", 4) if xbits >= 1 else ""
    qh2_body = _qh_block("qh2", 5) if xbits >= 2 else ""
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint row0 = tgid * (NSG * RS) + sg * RS;

    device const float4* x4 = (device const float4*)x;
    float acc[RS];
    for (uint j = 0; j < RS; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += 32) {
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float xbs = xbsum[p];
        for (uint j = 0; j < RS; j++) {
            uint4 qw = ((device const uint4*)(codes + (row0+j)*(IN_C/2)))[p];
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
""" + qh_body + qh2_body + """
            float dv = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                     + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                     + dot(xa2, float4(l2.x, l2.y, l2.z, l2.w))
                     + dot(xa3, float4(l3.x, l3.y, l3.z, l3.w))
                     + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                     + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w))
                     + dot(xb2, float4(h2.x, h2.y, h2.z, h2.w))
                     + dot(xb3, float4(h3.x, h3.y, h3.z, h3.w));
            uint sbi = (row0+j)*NSB + p/8;
            half  s  = (half)((float)qs[(row0+j)*NB + p] * (float)d[sbi]);
            half  mn = (half)((float)qm[(row0+j)*NB + p] * (float)dm[sbi]);
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
        name=f"k3_gw{4 + xbits}_s{NSG}r{RS}_{IN}_{OUT}",
        input_names=["x", "codes", "qs", "qm", "d", "dm", "qh", "qh2", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _cache[key] = kern
    return kern


class Syn:
    """Синтетический sb6-тензор + оба пути вызова."""

    def __init__(self, OUT, IN, xbits, rng):
        self.OUT, self.IN, self.xbits = OUT, IN, xbits
        NB, NSB = IN // 32, IN // 256
        self.NB, self.NSB = NB, NSB
        self.codes = mx.array(rng.integers(0, 256, (OUT, IN // 2), dtype=np.uint8))
        self.qs = mx.array(rng.integers(0, 64, (OUT, NB), dtype=np.uint8))
        self.qm = mx.array(rng.integers(-31, 32, (OUT, NB)).astype(np.int8))
        self.d = mx.array((rng.random((OUT, NSB), dtype=np.float32) * 1e-3 + 1e-3).astype(np.float16))
        self.dm = mx.array((rng.random((OUT, NSB), dtype=np.float32) * 1e-3 + 1e-3).astype(np.float16))
        self.qh = (mx.array(rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8))
                   if xbits >= 1 else mx.zeros((1,), dtype=mx.uint8))
        self.qh2 = (mx.array(rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8))
                    if xbits >= 2 else mx.zeros((1,), dtype=mx.uint8))
        mx.eval(self.codes, self.qs, self.qm, self.d, self.dm, self.qh, self.qh2)
        self.mb = (self.codes.size + self.qs.size + self.qm.size
                   + self.d.size * 2 + self.dm.size * 2
                   + (self.qh.size if xbits >= 1 else 0)
                   + (self.qh2.size if xbits >= 2 else 0)) / 1e6

    def old(self, x, xbsum):
        kern = _get_kernel_gw(self.IN, self.OUT, self.xbits)
        n_groups = (self.OUT + R - 1) // R
        return kern(
            inputs=[x, self.codes, self.qs, self.qm, self.d, self.dm,
                    self.qh, self.qh2, xbsum],
            grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(1, self.OUT)], output_dtypes=[mx.float32])[0]

    def k3(self, x, xbsum, NSG, RS):
        kern = get_kernel_k3(self.IN, self.OUT, self.xbits, NSG, RS)
        n_tg = self.OUT // (NSG * RS)
        return kern(
            inputs=[x, self.codes, self.qs, self.qm, self.d, self.dm,
                    self.qh, self.qh2, xbsum],
            grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
            output_shapes=[(1, self.OUT)], output_dtypes=[mx.float32])[0]


def bench(fn, reps=25, warm=4, calls=8):
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
    ("head5 65536x2048", 65536, 2048, 1),
]
VARIANTS = [(2, 4), (4, 4), (2, 8), (2, 2)]

rng = np.random.default_rng(0)
print("=== кернель-3 фаза B: реструктуризация (NSG simd x RS строк) ===")
for name, OUT, IN, xbits in CASES:
    t = Syn(OUT, IN, xbits, rng)
    x = mx.array(rng.standard_normal((1, IN)).astype(np.float32))
    xbsum = mx.sum(x.reshape(1, t.NB, 32), axis=2)
    mx.eval(x, xbsum)

    ref = np.array(t.old(x, xbsum))
    ok = {}
    for NSG, RS in VARIANTS:
        if OUT % (NSG * RS):
            continue
        got = np.array(t.k3(x, xbsum, NSG, RS))
        ok[(NSG, RS)] = bool(np.array_equal(ref, got))

    # A/B-чередование: old и все варианты по кругу, медиана по раундам
    res = {"old": []}
    for v in ok:
        res[v] = []
    for _ in range(3):  # 3 чередующихся мегараунда
        res["old"] += bench(lambda: t.old(x, xbsum), reps=8)
        for NSG, RS in ok:
            res[(NSG, RS)] += bench(lambda: t.k3(x, xbsum, NSG, RS), reps=8)
    mo = float(np.median(res["old"]))
    line = f"{name:18s} old={mo:6.3f}ms [{t.mb/mo:5.1f}GB/s]"
    for v in ok:
        m = float(np.median(res[v]))
        line += (f"  s{v[0]}r{v[1]}={m:6.3f} [{t.mb/m:5.1f}GB/s] "
                 f"x{mo/m:4.2f} {'BIT=' if ok[v] else 'DIFF!'}")
    print(line, flush=True)
    del t
