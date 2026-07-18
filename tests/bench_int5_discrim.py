"""Дискриминация лимитера int5: baseline / qh-load-no-insert (байты да,
ALU нет; численно НЕВЕРНО -- только полоса) / no-qh (int4-путь по тем же
codes; тоже неверно) на head. + int4 cmix key для сравнения абсолютов."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from bench_int5_variants import make_kernel  # baseline-строитель

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
head = model.head
key0 = model.blocks[0].cmix.key   # int4 8192x2048

def make_loadonly(IN, OUT, TG=32, R=8):
    NB, NSB = IN // 32, IN // 256
    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint TG    = {TG};
constant uint R     = {R};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
"""
    body = """
    uint g    = threadgroup_position_in_grid.x;
    uint n    = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint row0 = g * R;
    device const float4* x4 = (device const float4*)(x + n*IN_C);
    float acc[R];
    for (uint j = 0; j < R; j++) acc[j] = 0.0f;
    for (uint p = lane; p < NB; p += TG) {
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float xbs = xbsum[n*NB + p];
        for (uint j = 0; j < R; j++) {
            uint4 qw = ((device const uint4*)(codes + (row0+j)*(IN_C/2)))[p];
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
            uint hb = ((device const uint*)(qh + (row0+j)*(IN_C/8)))[p];
            float dv = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                     + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                     + dot(xa2, float4(l2.x, l2.y, l2.z, l2.w))
                     + dot(xa3, float4(l3.x, l3.y, l3.z, l3.w))
                     + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                     + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w))
                     + dot(xb2, float4(h2.x, h2.y, h2.z, h2.w))
                     + dot(xb3, float4(h3.x, h3.y, h3.z, h3.w))
                     + (float)(hb & 1u) * 1e-30f;
            uint sbi = (row0+j)*NSB + p/8;
            half  s  = (half)((float)qs[(row0+j)*NB + p] * (float)d[sbi]);
            half  mn = (half)((float)qm[(row0+j)*NB + p] * (float)dm[sbi]);
            acc[j] += (float)s * dv + (float)mn * xbs;
        }
    }
    for (uint j = 0; j < R; j++) {
        float a = simd_sum(acc[j]);
        if (lane == 0) out[n*OUT_C + (row0 + j)] = a;
    }
"""
    return mx.fast.metal_kernel(
        name=f"gw5_loadonly_{IN}_{OUT}",
        input_names=["x", "codes", "qs", "qm", "d", "dm", "qh", "xbsum"],
        output_names=["out"], header=hdr, source=body)

IN, OUT = head.in_features, head.out_features
x = mx.array(np.random.randn(1, IN).astype(np.float32))
xbsum = mx.sum(x.reshape(1, head.NB, 32), axis=2); mx.eval(x, xbsum)

k_base = make_kernel(IN, OUT, 32, 8, "insert", 1)
k_load = make_loadonly(IN, OUT)
import rwkv_quant.backends.metal.quant_linear_gw as gwmod
k_noqh = gwmod._get_kernel_gw(IN, OUT, has_qh=False)   # int4-семантика

ng = OUT // 8
def call(k):
    return k(inputs=[x, head.codes, head.qs, head.qm, head.d, head.dm,
                     head.qh, xbsum],
             grid=(ng * 32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(1, OUT)], output_dtypes=[mx.float32])[0]

MB5 = (head.codes.size + head.qs.size + head.qm.size + head.qh.size
       + head.d.size*2 + head.dm.size*2) / 1e6
MB4 = MB5 - head.qh.size / 1e6

cases = [("baseline int5", lambda: call(k_base), MB5),
         ("qh load, no insert", lambda: call(k_load), MB5),
         ("no qh at all (int4)", lambda: call(k_noqh), MB4)]
acc = {n: [] for n, *_ in cases}
for _ in range(5):
    for n, fn, _mb in cases:
        for _ in range(3): mx.eval(fn())
        mx.synchronize()
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
            ts.append(time.perf_counter() - t0)
        acc[n].append(np.median(ts)*1e3)
for n, fn, mb in cases:
    t = np.median(acc[n])
    print(f"{n:22s} {t:7.3f} ms  {mb/t:6.1f} GB/s ({mb:.1f}MB)")

# int4 cmix key прямым боевым кернелем (для абсолюта int4 gw)
xk = mx.array(np.random.randn(1, 2048).astype(np.float32)); mx.eval(xk)
for _ in range(5): mx.eval(key0(xk))
mx.synchronize()
ts = []
for _ in range(40):
    t0 = time.perf_counter(); mx.eval(key0(xk)); mx.synchronize()
    ts.append(time.perf_counter()-t0)
MBk = (key0.codes.size + key0.qs.size + key0.qm.size + key0.d.size*2 + key0.dm.size*2)/1e6
t = np.median(ts)*1e3
print(f"{'int4 key 8192x2048':22s} {t:7.3f} ms  {MBk/t:6.1f} GB/s ({MBk:.1f}MB, launch не вычтен)")
