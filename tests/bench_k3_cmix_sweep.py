"""Свип cmix-форм (int4/int6): расширенные (NSG,RS) + packs2 (поток берёт
2 СМЕЖНЫХ блока: 32Б контигуозного кода на строку, x-регистры x2, RS вдвое
меньше). A/B в одном процессе. Цель: 75 -> 90+ GB/s."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.backends.metal.quant_linear_gw import _get_kernel_k3, _k3_cfg

_cache = {}
M = "0x00204081u & 0x01010101u"

def _plane_p2(idx, shift, qb):
    return f"""
            uint hbA{idx} = {qb}[{4+idx}];
            l0 |= as_type<uchar4>((( hbA{idx}        & 0xFu) * {M}) << {shift});
            l1 |= as_type<uchar4>((((hbA{idx} >>  4) & 0xFu) * {M}) << {shift});
            l2 |= as_type<uchar4>((((hbA{idx} >>  8) & 0xFu) * {M}) << {shift});
            l3 |= as_type<uchar4>((((hbA{idx} >> 12) & 0xFu) * {M}) << {shift});
            h0 |= as_type<uchar4>((((hbA{idx} >> 16) & 0xFu) * {M}) << {shift});
            h1 |= as_type<uchar4>((((hbA{idx} >> 20) & 0xFu) * {M}) << {shift});
            h2 |= as_type<uchar4>((((hbA{idx} >> 24) & 0xFu) * {M}) << {shift});
            h3 |= as_type<uchar4>((( hbA{idx} >> 28)         * {M}) << {shift});
"""

def get_kernel_p2(IN, OUT, xbits, NSG, RS):
    """packs2: p-шаг 2 блока; поток обрабатывает блоки 2q и 2q+1."""
    key = ("p2", IN, OUT, xbits, NSG, RS)
    if key in _cache: return _cache[key]
    NB, NSB = IN // 32, IN // 256
    SU = 4 + xbits
    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
constant uint NSG   = {NSG};
constant uint RS    = {RS};
constant uint SU    = {SU};
"""
    dec = ""
    if xbits >= 1: dec += _plane_p2(0, 4, "qbA") + _plane_p2(0, 4, "qbB").replace("hbA0", "hbB0").replace("l0", "m0").replace("l1", "m1").replace("l2", "m2").replace("l3", "m3").replace("h0", "g0").replace("h1", "g1").replace("h2", "g2").replace("h3", "g3")
    if xbits >= 2: dec += _plane_p2(1, 5, "qbA") + _plane_p2(1, 5, "qbB").replace("hbA1", "hbB1").replace("l0", "m0").replace("l1", "m1").replace("l2", "m2").replace("l3", "m3").replace("h0", "g0").replace("h1", "g1").replace("h2", "g2").replace("h3", "g3")
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint row0 = tgid * (NSG * RS) + sg * RS;

    device const float4* x4  = (device const float4*)x;
    device const uint*   qu  = (device const uint*)qblk;
    device const uchar2* sm2 = (device const uchar2*)qsqm;
    device const half2*  dd2 = (device const half2*)ddm;
    float acc[RS];
    for (uint j = 0; j < RS; j++) acc[j] = 0.0f;

    for (uint q = lane; q < NB/2; q += 32) {
        uint p = q * 2;
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float4 ya0 = x4[p*8+8], ya1 = x4[p*8+9], ya2 = x4[p*8+10], ya3 = x4[p*8+11];
        float4 yb0 = x4[p*8+12], yb1 = x4[p*8+13], yb2 = x4[p*8+14], yb3 = x4[p*8+15];
        float xbsA = xbsum[p], xbsB = xbsum[p+1];
        for (uint j = 0; j < RS; j++) {
            device const uint* qbA = qu + ((row0+j)*NB + p) * SU;
            device const uint* qbB = qbA + SU;
            uint4 qwA = uint4(qbA[0], qbA[1], qbA[2], qbA[3]);
            uint4 qwB = uint4(qbB[0], qbB[1], qbB[2], qbB[3]);
            uchar4 l0 = as_type<uchar4>(qwA.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qwA.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qwA.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qwA.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qwA.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qwA.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qwA.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qwA.w >> 4) & 0x0F0F0F0Fu);
            uchar4 m0 = as_type<uchar4>(qwB.x & 0x0F0F0F0Fu);
            uchar4 m1 = as_type<uchar4>(qwB.y & 0x0F0F0F0Fu);
            uchar4 m2 = as_type<uchar4>(qwB.z & 0x0F0F0F0Fu);
            uchar4 m3 = as_type<uchar4>(qwB.w & 0x0F0F0F0Fu);
            uchar4 g0 = as_type<uchar4>((qwB.x >> 4) & 0x0F0F0F0Fu);
            uchar4 g1 = as_type<uchar4>((qwB.y >> 4) & 0x0F0F0F0Fu);
            uchar4 g2 = as_type<uchar4>((qwB.z >> 4) & 0x0F0F0F0Fu);
            uchar4 g3 = as_type<uchar4>((qwB.w >> 4) & 0x0F0F0F0Fu);
""" + dec + """
            float dvA = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                      + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                      + dot(xa2, float4(l2.x, l2.y, l2.z, l2.w))
                      + dot(xa3, float4(l3.x, l3.y, l3.z, l3.w))
                      + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                      + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w))
                      + dot(xb2, float4(h2.x, h2.y, h2.z, h2.w))
                      + dot(xb3, float4(h3.x, h3.y, h3.z, h3.w));
            float dvB = dot(ya0, float4(m0.x, m0.y, m0.z, m0.w))
                      + dot(ya1, float4(m1.x, m1.y, m1.z, m1.w))
                      + dot(ya2, float4(m2.x, m2.y, m2.z, m2.w))
                      + dot(ya3, float4(m3.x, m3.y, m3.z, m3.w))
                      + dot(yb0, float4(g0.x, g0.y, g0.z, g0.w))
                      + dot(yb1, float4(g1.x, g1.y, g1.z, g1.w))
                      + dot(yb2, float4(g2.x, g2.y, g2.z, g2.w))
                      + dot(yb3, float4(g3.x, g3.y, g3.z, g3.w));
            uchar2 smA = sm2[(row0+j)*NB + p];
            uchar2 smB = sm2[(row0+j)*NB + p + 1];
            half2  dd = dd2[(row0+j)*NSB + p/8];
            half  sA  = (half)((float)smA.x * (float)dd.x);
            half  mnA = (half)((float)as_type<char>(smA.y) * (float)dd.y);
            half  sB  = (half)((float)smB.x * (float)dd.x);
            half  mnB = (half)((float)as_type<char>(smB.y) * (float)dd.y);
            acc[j] += (float)sA * dvA + (float)mnA * xbsA
                    + (float)sB * dvB + (float)mnB * xbsB;
        }
    }
    for (uint j = 0; j < RS; j++) {
        float a = simd_sum(acc[j]);
        if (lane == 0)
            out[row0 + j] = a;
    }
"""
    kern = mx.fast.metal_kernel(
        name=f"k3p2_gw{4+xbits}_s{NSG}r{RS}_{IN}_{OUT}",
        input_names=["x", "qblk", "qsqm", "ddm", "xbsum"],
        output_names=["out"], header=hdr, source=body)
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
        d = (rng.random((OUT, NSB), dtype=np.float32)*1e-3+1e-3).astype(np.float16)
        dm = (rng.random((OUT, NSB), dtype=np.float32)*1e-3+1e-3).astype(np.float16)
        qh = rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8)
        qh2 = rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8)
        parts = [codes.reshape(OUT, NB, 16)]
        if xbits >= 1: parts.append(qh.reshape(OUT, NB, 4))
        if xbits >= 2: parts.append(qh2.reshape(OUT, NB, 4))
        self.qblk = mx.array(np.ascontiguousarray(
            np.concatenate(parts, axis=2).reshape(OUT, -1)))
        self.qsqm = mx.array(np.ascontiguousarray(
            np.stack([qs, qm.view(np.uint8)], axis=-1).reshape(OUT, -1)))
        self.ddm = mx.array(np.ascontiguousarray(
            np.stack([d, dm], axis=-1).reshape(OUT, -1)))
        mx.eval(self.qblk, self.qsqm, self.ddm)
        self.mb = (codes.size + qs.size + qm.size + d.size*2 + dm.size*2
                   + (qh.size if xbits >= 1 else 0)
                   + (qh2.size if xbits >= 2 else 0)) / 1e6

    def k3(self, x, xbsum, NSG=0, RS=0):
        if not NSG:
            NSG, RS = _k3_cfg(self.IN, self.OUT, self.xbits)
        kern = _get_kernel_k3(self.IN, self.OUT, self.xbits, NSG, RS)
        n_tg = self.OUT // (NSG * RS)
        return kern(inputs=[x, self.qblk, self.qsqm, self.ddm, xbsum],
                    grid=(n_tg*NSG*32, 1, 1), threadgroup=(NSG*32, 1, 1),
                    output_shapes=[(1, self.OUT)], output_dtypes=[mx.float32])[0]

    def p2(self, x, xbsum, NSG, RS):
        kern = get_kernel_p2(self.IN, self.OUT, self.xbits, NSG, RS)
        n_tg = self.OUT // (NSG * RS)
        return kern(inputs=[x, self.qblk, self.qsqm, self.ddm, xbsum],
                    grid=(n_tg*NSG*32, 1, 1), threadgroup=(NSG*32, 1, 1),
                    output_shapes=[(1, self.OUT)], output_dtypes=[mx.float32])[0]


def bench(fn, reps=8, warm=4, calls=8):
    for _ in range(warm): mx.eval(*[fn() for _ in range(calls)])
    mx.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn() for _ in range(calls)])
        mx.synchronize(); ts.append((time.perf_counter()-t0)*1e3/calls)
    return ts


rng = np.random.default_rng(0)
CASES = [("cmixK4 8192x2048", 8192, 2048, 0),
         ("cmixV4 2048x8192", 2048, 8192, 0),
         ("cmixK6 8192x2048", 8192, 2048, 2),
         ("cmixV6 2048x8192", 2048, 8192, 2)]
CFG_K3 = [(2, 2), (8, 2), (8, 4), (4, 8)]
CFG_P2 = [(2, 2), (4, 2), (2, 4), (4, 4)]

print("=== cmix-свип: k3-конфиги + packs2 ===")
for name, OUT, IN, xbits in CASES:
    t = Syn(OUT, IN, xbits, rng)
    x = mx.array(rng.standard_normal((1, IN)).astype(np.float32))
    xbsum = mx.sum(x.reshape(1, t.NB, 32), axis=2); mx.eval(x, xbsum)
    ref = np.array(t.k3(x, xbsum)); dmax = np.abs(ref).max() + 1e-6
    variants = {}
    for c in CFG_K3:
        if OUT % (c[0]*c[1]) == 0:
            variants[("k", c)] = lambda c=c: t.k3(x, xbsum, *c)
    for c in CFG_P2:
        if OUT % (c[0]*c[1]) == 0:
            d0 = float(np.max(np.abs(np.array(t.p2(x, xbsum, *c)) - ref)) / dmax)
            if d0 > 1e-3:
                print(f"  !! p2 {c} diff {d0:.1e}")
            variants[("p", c)] = lambda c=c: t.p2(x, xbsum, *c)
    res = {"base": []}
    for v in variants: res[v] = []
    for _ in range(3):
        res["base"] += bench(lambda: t.k3(x, xbsum))
        for v in variants: res[v] += bench(variants[v])
    mo = float(np.median(res["base"]))
    line = f"{name:16s} base={mo:6.3f} [{t.mb/mo:5.1f}]"
    for v in variants:
        m = float(np.median(res[v]))
        line += f"  {v[0]}{v[1][0]}x{v[1][1]}={m:6.3f} [{t.mb/m:5.1f}] x{mo/m:4.2f}"
    print(line, flush=True)
    del t
