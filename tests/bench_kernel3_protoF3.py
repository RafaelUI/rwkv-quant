"""Очный A/B полублоков в ОДНОМ процессе (закон 1): k3 (боевой) vs
полублок-ПЕРМ (ushort-лоады, байты плоскостей [b0,b2,b1,b3]) vs
полублок-БЕЗ-перма динамический сдвиг vs БЕЗ-перма статические ветки.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.backends.metal.quant_linear_gw import (
    _get_kernel_k3, _k3_cfg)

_cache = {}
M = "0x00204081u & 0x01010101u"


def _mk_plane(mode, buf, idx, shift):
    if mode == "perm":
        return f"""
            ushort hs{idx} = *(device const ushort*)((device const uchar*)(qb + {idx}) + off2);
            l0 |= as_type<uchar4>((((uint)(hs{idx} >>  0) & 0xFu) * {M}) << {shift});
            l1 |= as_type<uchar4>((((uint)(hs{idx} >>  4) & 0xFu) * {M}) << {shift});
            h0 |= as_type<uchar4>((((uint)(hs{idx} >>  8) & 0xFu) * {M}) << {shift});
            h1 |= as_type<uchar4>((((uint)(hs{idx} >> 12)       ) * {M}) << {shift});
"""
    if mode == "dyn":
        return f"""
            uint hb{idx} = qb[{idx}];
            uint s{idx} = half_i * 8;
            l0 |= as_type<uchar4>((((hb{idx} >> (s{idx}+ 0)) & 0xFu) * {M}) << {shift});
            l1 |= as_type<uchar4>((((hb{idx} >> (s{idx}+ 4)) & 0xFu) * {M}) << {shift});
            h0 |= as_type<uchar4>((((hb{idx} >> (s{idx}+16)) & 0xFu) * {M}) << {shift});
            h1 |= as_type<uchar4>((((hb{idx} >> (s{idx}+20)) & 0xFu) * {M}) << {shift});
"""
    # static: две компайл-тайм ветки
    return f"""
            uint hb{idx} = qb[{idx}];
            if (half_i) {{
                l0 |= as_type<uchar4>((((hb{idx} >>  8) & 0xFu) * {M}) << {shift});
                l1 |= as_type<uchar4>((((hb{idx} >> 12) & 0xFu) * {M}) << {shift});
                h0 |= as_type<uchar4>((((hb{idx} >> 24) & 0xFu) * {M}) << {shift});
                h1 |= as_type<uchar4>((( hb{idx} >> 28)        * {M}) << {shift});
            }} else {{
                l0 |= as_type<uchar4>((( hb{idx}        & 0xFu) * {M}) << {shift});
                l1 |= as_type<uchar4>((((hb{idx} >>  4) & 0xFu) * {M}) << {shift});
                h0 |= as_type<uchar4>((((hb{idx} >> 16) & 0xFu) * {M}) << {shift});
                h1 |= as_type<uchar4>((((hb{idx} >> 20) & 0xFu) * {M}) << {shift});
            }}
"""


def get_kernel_half(IN, OUT, xbits, NSG, RS, mode):
    key = (IN, OUT, xbits, NSG, RS, mode)
    if key in _cache:
        return _cache[key]
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
    if xbits >= 1:
        dec += _mk_plane(mode, "qb", 4, 4)
    if xbits >= 2:
        dec += _mk_plane(mode, "qb", 5, 5)
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint half_i = lane & 1;
    uint off2   = half_i * 2;
    uint row0 = tgid * (NSG * RS) + sg * RS;
    (void)off2;

    device const float4* x4  = (device const float4*)x;
    device const uint*   qu  = (device const uint*)qblk;
    device const uchar2* sm2 = (device const uchar2*)qsqm;
    device const half2*  dd2 = (device const half2*)ddm;
    float acc[RS];
    for (uint j = 0; j < RS; j++) acc[j] = 0.0f;

    for (uint p = lane >> 1; p < NB; p += 16) {
        float4 xa0 = x4[p*8 + half_i*2 + 0], xa1 = x4[p*8 + half_i*2 + 1];
        float4 xb0 = x4[p*8 + half_i*2 + 4], xb1 = x4[p*8 + half_i*2 + 5];
        float xbs = half_i ? 0.0f : xbsum[p];
        for (uint j = 0; j < RS; j++) {
            device const uint* qb = qu + ((row0+j)*NB + p) * SU;
            uint2 qw = *(device const uint2*)(qb + half_i*2);
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
""" + dec + """
            float dv = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                     + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                     + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                     + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w));
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
        name=f"k3h_{mode}_gw{4 + xbits}_s{NSG}r{RS}_{IN}_{OUT}",
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
        parts = [codes.reshape(OUT, NB, 16)]
        if xbits >= 1: parts.append(qh.reshape(OUT, NB, 4))
        if xbits >= 2: parts.append(qh2.reshape(OUT, NB, 4))
        self.qblk = mx.array(np.ascontiguousarray(
            np.concatenate(parts, axis=2).reshape(OUT, -1)))
        perm = [0, 2, 1, 3]
        partsP = [codes.reshape(OUT, NB, 16)]
        if xbits >= 1: partsP.append(qh.reshape(OUT, NB, 4)[:, :, perm])
        if xbits >= 2: partsP.append(qh2.reshape(OUT, NB, 4)[:, :, perm])
        self.qblkP = mx.array(np.ascontiguousarray(
            np.concatenate(partsP, axis=2).reshape(OUT, -1)))
        self.qsqm = mx.array(np.ascontiguousarray(
            np.stack([qs, qm.view(np.uint8)], axis=-1).reshape(OUT, -1)))
        self.ddm = mx.array(np.ascontiguousarray(
            np.stack([d, dm], axis=-1).reshape(OUT, -1)))
        mx.eval(self.qblk, self.qblkP, self.qsqm, self.ddm)
        self.mb = (codes.size + qs.size + qm.size + d.size*2 + dm.size*2
                   + (qh.size if xbits >= 1 else 0)
                   + (qh2.size if xbits >= 2 else 0)) / 1e6

    def k3(self, x, xbsum):
        NSG, RS = _k3_cfg(self.IN, self.OUT, self.xbits)
        kern = _get_kernel_k3(self.IN, self.OUT, self.xbits, NSG, RS)
        n_tg = self.OUT // (NSG * RS)
        return kern(inputs=[x, self.qblk, self.qsqm, self.ddm, xbsum],
                    grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                    output_shapes=[(1, self.OUT)], output_dtypes=[mx.float32])[0]

    def half(self, x, xbsum, NSG, RS, mode):
        kern = get_kernel_half(self.IN, self.OUT, self.xbits, NSG, RS, mode)
        qb = self.qblkP if mode == "perm" else self.qblk
        n_tg = self.OUT // (NSG * RS)
        return kern(inputs=[x, qb, self.qsqm, self.ddm, xbsum],
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
    ("tmix5 2048x2048",  2048, 2048, 1, (4, 8)),
    ("tmix6 2048x2048",  2048, 2048, 2, (4, 4)),
    ("cmixK6 8192x2048", 8192, 2048, 2, (2, 4)),
    ("cmixV6 2048x8192", 2048, 8192, 2, (2, 4)),
    ("head5 65536x2048", 65536, 2048, 1, (2, 4)),
    ("head6 65536x2048", 65536, 2048, 2, (2, 4)),
]
MODES = ["perm", "dyn", "static"]

rng = np.random.default_rng(0)
print("=== очный A/B: k3 vs полублок perm/dyn/static ===")
for name, OUT, IN, xbits, (NSG, RS) in CASES:
    t = Syn(OUT, IN, xbits, rng)
    x = mx.array(rng.standard_normal((1, IN)).astype(np.float32))
    xbsum = mx.sum(x.reshape(1, t.NB, 32), axis=2)
    mx.eval(x, xbsum)
    ref = np.array(t.k3(x, xbsum))
    dmax = np.abs(ref).max() + 1e-6
    diffs = {m: float(np.max(np.abs(np.array(t.half(x, xbsum, NSG, RS, m)) - ref)) / dmax)
             for m in MODES}
    res = {"k3": []}
    for m in MODES:
        res[m] = []
    for _ in range(3):
        res["k3"] += bench(lambda: t.k3(x, xbsum))
        for m in MODES:
            res[m] += bench(lambda: t.half(x, xbsum, NSG, RS, m))
    mo = float(np.median(res["k3"]))
    line = f"{name:18s} s{NSG}r{RS} k3={mo:6.3f} [{t.mb/mo:5.1f}GB/s]"
    for m in MODES:
        v = float(np.median(res[m]))
        ok = "ok" if diffs[m] < 1e-3 else f"REL{diffs[m]:.0e}"
        line += f"  {m}={v:6.3f} [{t.mb/v:5.1f}] x{mo/v:4.2f} {ok}"
    print(line, flush=True)
    del t
