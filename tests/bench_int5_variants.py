"""Варианты int5 gw-кернеля: baseline / qh-через-dots / R∈{4,16} / unroll2.
Формы: head 65536x2048 (чистая полоса) + receptance 2048x2048 (боевая
мелкая). Гейт: relmax vs текущий кернель. Round-robin чередование."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
import rwkv_quant.backends.metal.quant_linear_gw as gw

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
head = model.head
rec = model.blocks[0].tmix.r_proj
assert head.has_qh and rec.has_qh

def make_kernel(IN, OUT, TG, R, qh_mode, unroll):
    NB, NSB = IN // 32, IN // 256
    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint TG    = {TG};
constant uint R     = {R};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
"""
    qh_insert = """
            uint hb = ((device const uint*)(qh + (row0+j)*(IN_C/8)))[P];
            l0 |= uchar4((uint4(hb) >> uint4( 0, 1, 2, 3)) & 1u) << 4;
            l1 |= uchar4((uint4(hb) >> uint4( 4, 5, 6, 7)) & 1u) << 4;
            l2 |= uchar4((uint4(hb) >> uint4( 8, 9,10,11)) & 1u) << 4;
            l3 |= uchar4((uint4(hb) >> uint4(12,13,14,15)) & 1u) << 4;
            h0 |= uchar4((uint4(hb) >> uint4(16,17,18,19)) & 1u) << 4;
            h1 |= uchar4((uint4(hb) >> uint4(20,21,22,23)) & 1u) << 4;
            h2 |= uchar4((uint4(hb) >> uint4(24,25,26,27)) & 1u) << 4;
            h3 |= uchar4((uint4(hb) >> uint4(28,29,30,31)) & 1u) << 4;
"""
    blk = """
        {
        uint P = PEXPR;
        float4 xa0 = x4[P*8+0], xa1 = x4[P*8+1], xa2 = x4[P*8+2], xa3 = x4[P*8+3];
        float4 xb0 = x4[P*8+4], xb1 = x4[P*8+5], xb2 = x4[P*8+6], xb3 = x4[P*8+7];
        float xbs = xbsum[n*NB + P];
        for (uint j = 0; j < R; j++) {
            uint4 qw = ((device const uint4*)(codes + (row0+j)*(IN_C/2)))[P];
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
QH_INSERT
            float dv = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                     + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                     + dot(xa2, float4(l2.x, l2.y, l2.z, l2.w))
                     + dot(xa3, float4(l3.x, l3.y, l3.z, l3.w))
                     + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                     + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w))
                     + dot(xb2, float4(h2.x, h2.y, h2.z, h2.w))
                     + dot(xb3, float4(h3.x, h3.y, h3.z, h3.w));
QH_DOTS
            uint sbi = (row0+j)*NSB + P/8;
            half  s  = (half)((float)qs[(row0+j)*NB + P] * (float)d[sbi]);
            half  mn = (half)((float)qm[(row0+j)*NB + P] * (float)dm[sbi]);
            acc[j] += (float)s * dv + (float)mn * xbs;
        }
        }
"""
    qh_dots = """
            uint hb = ((device const uint*)(qh + (row0+j)*(IN_C/8)))[P];
            dv += 16.0f * (
                  dot(xa0, float4(uint4(hb) >> uint4( 0, 1, 2, 3) & 1u))
                + dot(xa1, float4(uint4(hb) >> uint4( 4, 5, 6, 7) & 1u))
                + dot(xa2, float4(uint4(hb) >> uint4( 8, 9,10,11) & 1u))
                + dot(xa3, float4(uint4(hb) >> uint4(12,13,14,15) & 1u))
                + dot(xb0, float4(uint4(hb) >> uint4(16,17,18,19) & 1u))
                + dot(xb1, float4(uint4(hb) >> uint4(20,21,22,23) & 1u))
                + dot(xb2, float4(uint4(hb) >> uint4(24,25,26,27) & 1u))
                + dot(xb3, float4(uint4(hb) >> uint4(28,29,30,31) & 1u)));
"""
    if qh_mode == "insert":
        blk = blk.replace("QH_INSERT", qh_insert).replace("QH_DOTS", "")
    else:
        blk = blk.replace("QH_INSERT", "").replace("QH_DOTS", qh_dots)

    if unroll == 1:
        loop = ('    for (uint p = lane; p < NB; p += TG) {\n'
                + blk.replace("PEXPR", "p") + '    }\n')
    else:
        loop = ('    for (uint p = lane; p < NB; p += 2*TG) {\n'
                + blk.replace("PEXPR", "p")
                + '        if (p + TG < NB) {\n'
                + blk.replace("PEXPR", "p + TG")
                + '        }\n    }\n')

    body = """
    uint g    = threadgroup_position_in_grid.x;
    uint n    = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint row0 = g * R;

    device const float4* x4 = (device const float4*)(x + n*IN_C);
    float acc[R];
    for (uint j = 0; j < R; j++) acc[j] = 0.0f;
""" + loop + """
    for (uint j = 0; j < R; j++) {
        float a = simd_sum(acc[j]);
        if (lane == 0)
            out[n*OUT_C + (row0 + j)] = a;
    }
"""
    return mx.fast.metal_kernel(
        name=f"gw5var_{IN}_{OUT}_{TG}_{R}_{qh_mode}_{unroll}",
        input_names=["x", "codes", "qs", "qm", "d", "dm", "qh", "xbsum"],
        output_names=["out"], header=hdr, source=body)

def run_variant(lin, TG, R, qh_mode, unroll, x, xbsum):
    OUT, IN = lin.out_features, lin.in_features
    kern = make_kernel(IN, OUT, TG, R, qh_mode, unroll)
    n_groups = (OUT + R - 1) // R
    def call():
        return kern(inputs=[x, lin.codes, lin.qs, lin.qm, lin.d, lin.dm,
                            lin.qh, xbsum],
                    grid=(n_groups * TG, 1, 1), threadgroup=(TG, 1, 1),
                    output_shapes=[(1, OUT)], output_dtypes=[mx.float32])[0]
    return call

VARIANTS = [("baseline R8", 32, 8, "insert", 1),
            ("qh-dots R8",  32, 8, "dots",   1),
            ("R4",          32, 4, "insert", 1),
            ("R16",         32, 16, "insert", 1),
            ("unroll2 R8",  32, 8, "insert", 2),
            ("qh-dots un2", 32, 8, "dots",   2)]

for name, lin in (("head 65536x2048", head), ("rec 2048x2048", rec)):
    IN = lin.in_features
    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    xbsum = mx.sum(x.reshape(1, lin.NB, 32), axis=2)
    mx.eval(x, xbsum)
    ref = lin(x); mx.eval(ref)   # текущий боевой кернель
    calls, ok = {}, {}
    for vn, TG, R, qm_, un in VARIANTS:
        c = run_variant(lin, TG, R, qm_, un, x, xbsum)
        y = c(); mx.eval(y)
        rel = float(mx.max(mx.abs(y - ref)) / (mx.max(mx.abs(ref)) + 1e-9))
        calls[vn], ok[vn] = c, rel
    acc = {vn: [] for vn, *_ in VARIANTS}
    for _ in range(5):
        for vn, *_ in VARIANTS:
            c = calls[vn]
            for _ in range(3): mx.eval(c())
            mx.synchronize()
            ts = []
            for _ in range(15):
                t0 = time.perf_counter(); mx.eval(c()); mx.synchronize()
                ts.append(time.perf_counter() - t0)
            acc[vn].append(np.median(ts) * 1e3)
    MB = (lin.codes.size + lin.qs.size + lin.qm.size + lin.qh.size
          + lin.d.size * 2 + lin.dm.size * 2) / 1e6
    print(f"-- {name} ({MB:.1f}MB)")
    for vn, *_ in VARIANTS:
        t = np.median(acc[vn])
        print(f"   {vn:14s} {t:7.3f} ms  {MB/t:6.1f} GB/s  relmax={ok[vn]:.1e}")
