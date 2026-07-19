"""Кернель-3 фаза E: N-батчевый режим (NN=4, verify-путь спекулятивки).
Раскладка фазы D (NSG simdgroups x RS строк, интерлив qblk, мульт-трюк
битплоскостей), веса блока декодируются один раз на NN колонок.
Порядок математики на пару (строка, колонка) как у старого nb-кернеля.
A/B в одном процессе vs _get_kernel_gw_nb с боевым rb по форме.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.backends.metal.quant_linear_gw import (
    _get_kernel_gw_nb, TG, _RB_FOR_NN, RB_NB)

_cache = {}

MULT = "0x00204081u & 0x01010101u"


def _plane(src_idx, shift):
    ls = []
    for i, reg in enumerate(["l0", "l1", "l2", "l3", "h0", "h1", "h2", "h3"]):
        sh = i * 4
        nib = f"((hbx{src_idx} >> {sh:2d}) & 0xFu)" if sh else f"(hbx{src_idx} & 0xFu)"
        if sh == 28:
            nib = f"(hbx{src_idx} >> 28)"
        ls.append(f"            {reg} |= as_type<uchar4>(({nib} * {MULT}) << {shift});")
    return "\n".join(ls)


def get_kernel_k3nb(IN, OUT, xbits, NN, NSG=2, RS=4):
    key = (IN, OUT, xbits, NN, NSG, RS)
    if key in _cache:
        return _cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
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
constant uint NN    = {NN};
"""
    hb_load = "            uint hbx0 = qb[4];\n" + _plane(0, 4) if xbits >= 1 else ""
    hb2_load = "            uint hbx1 = qb[5];\n" + _plane(1, 5) if xbits >= 2 else ""
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
            uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
""" + hb_load + "\n" + hb2_load + """
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

    def _rb(self):
        if self.IN >= 8192: return 8
        if self.OUT >= 8192: return 4
        return _RB_FOR_NN.get(4, RB_NB)

    def old_nb(self, x, xbsum, NN):
        rb = self._rb()
        kern = _get_kernel_gw_nb(self.IN, self.OUT, self.xbits, NN, rb)
        n_groups = (self.OUT + rb - 1) // rb
        return kern(
            inputs=[x, self.codes, self.qs, self.qm, self.d, self.dm,
                    self.qh, self.qh2, xbsum],
            grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(NN, self.OUT)], output_dtypes=[mx.float32])[0]

    def k3nb(self, x, xbsum, NN, NSG, RS):
        kern = get_kernel_k3nb(self.IN, self.OUT, self.xbits, NN, NSG, RS)
        n_tg = self.OUT // (NSG * RS)
        return kern(
            inputs=[x, self.qblk, self.qsqm, self.ddm, xbsum],
            grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
            output_shapes=[(NN, self.OUT)], output_dtypes=[mx.float32])[0]


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


NN = 4
CASES = [
    ("tmix5 2048x2048",  2048, 2048, 1),
    ("tmix6 2048x2048",  2048, 2048, 2),
    ("cmixK4 8192x2048", 8192, 2048, 0),
    ("cmixK6 8192x2048", 8192, 2048, 2),
    ("cmixV4 2048x8192", 2048, 8192, 0),
    ("cmixV6 2048x8192", 2048, 8192, 2),
    ("head5 65536x2048", 65536, 2048, 1),
]
VARIANTS = [(2, 2), (2, 4), (4, 4), (4, 2)]

rng = np.random.default_rng(0)
print(f"=== кернель-3 фаза E: N-батч NN={NN} (verify-путь) ===")
for name, OUT, IN, xbits in CASES:
    t = Syn(OUT, IN, xbits, rng)
    x = mx.array(rng.standard_normal((NN, IN)).astype(np.float32))
    xbsum = mx.sum(x.reshape(NN, t.NB, 32), axis=2)
    mx.eval(x, xbsum)

    ref = np.array(t.old_nb(x, xbsum, NN))
    denom = np.maximum(np.abs(ref), 1e-6)
    diffs = {}
    for NSG, RS in VARIANTS:
        if OUT % (NSG * RS):
            continue
        got = np.array(t.k3nb(x, xbsum, NN, NSG, RS))
        diffs[(NSG, RS)] = float(np.max(np.abs(got - ref) / denom))

    res = {"old": []}
    for v in diffs:
        res[v] = []
    for _ in range(3):
        res["old"] += bench(lambda: t.old_nb(x, xbsum, NN))
        for v in diffs:
            res[v] += bench(lambda: t.k3nb(x, xbsum, NN, *v))
    mo = float(np.median(res["old"]))
    line = f"{name:18s} old(rb{t._rb()})={mo:6.3f}ms [{t.mb/mo:5.1f}GB/s]"
    for v in diffs:
        m = float(np.median(res[v]))
        tag = "BIT=" if diffs[v] == 0.0 else f"rd{diffs[v]:.0e}"
        line += f"  s{v[0]}r{v[1]}={m:6.3f} [{t.mb/m:5.1f}] x{mo/m:4.2f} {tag}"
    print(line, flush=True)
    del t
