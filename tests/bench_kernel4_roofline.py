"""Кернель-4 руфлайн-проба: где сидят 25% разрыва с MLX на cmix-формах.
Варианты одной формы (доступ к памяти идентичен):
  mem    -- все загрузки живы, декод/доты заменены дешёвым поглощением;
  nodot  -- полный декод весов, но без 8 dot'ов (только поглощение w);
  full   -- боевой к3 (импорт);
  premul -- qdot в стиле MLX: x пре-домножен на 2^-4k (точно, степени
            двойки), нибблы извлекаются масками БЕЗ сдвигов, без
            uchar4-каскада. Для int6 плоскости остаются мульт-трюком.
A/B в одном процессе, машина остывшая.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.backends.metal.quant_linear_gw import _get_kernel_k3, _k3_cfg

_cache = {}
M = "0x00204081u & 0x01010101u"

HDR = """
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
constant uint NSG   = {NSG};
constant uint RS    = {RS};
constant uint SU    = {SU};
"""

PRE = """
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
"""

POST = """
    for (uint j = 0; j < RS; j++) {
        float a = simd_sum(acc[j]);
        if (lane == 0)
            out[row0 + j] = a;
    }
"""

QLOADS = """
            device const uint* qb = qu + ((row0+j)*NB + p) * SU;
            uchar2 sm = sm2[(row0+j)*NB + p];
            half2  dd = dd2[(row0+j)*NSB + p/8];
            half  s  = (half)((float)sm.x * (float)dd.x);
            half  mn = (half)((float)as_type<char>(sm.y) * (float)dd.y);
"""


def make(IN, OUT, xbits, NSG, RS, mode):
    key = (IN, OUT, xbits, NSG, RS, mode)
    if key in _cache:
        return _cache[key]
    NB, NSB = IN // 32, IN // 256
    SU = 4 + xbits
    hdr = HDR.format(IN=IN, OUT=OUT, NB=NB, NSB=NSB, NSG=NSG, RS=RS, SU=SU)

    if mode == "mem":
        planes = "            uint hb = qb[4]; uint hb2 = qb[5];\n" if xbits >= 2 else (
                 "            uint hb = qb[4]; uint hb2 = 0u;\n" if xbits >= 1 else
                 "            uint hb = 0u; uint hb2 = 0u;\n")
        body = PRE + """
    for (uint p = lane; p < NB; p += 32) {
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float xbs = xbsum[p];
        for (uint j = 0; j < RS; j++) {
""" + QLOADS + planes + """
            uint mix = qb[0] ^ qb[1] ^ qb[2] ^ qb[3] ^ hb ^ hb2;
            acc[j] += (float)s * (float)(mix & 0xFFu)
                    * (xa0.x + xa1.y + xa2.z + xa3.w
                       + xb0.x + xb1.y + xb2.z + xb3.w)
                    + (float)mn * xbs;
        }
    }
""" + POST
    elif mode == "nodot":
        dec = ""
        if xbits >= 1:
            dec += "            uint hb = qb[4];\n" + _mk_mult("hb", 4)
        if xbits >= 2:
            dec += "            uint hb2 = qb[5];\n" + _mk_mult("hb2", 5)
        body = PRE + """
    for (uint p = lane; p < NB; p += 32) {
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float xbs = xbsum[p];
        for (uint j = 0; j < RS; j++) {
""" + QLOADS + """
            uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
""" + dec + """
            acc[j] += (float)s * (float)(l0.x + l1.y + l2.z + l3.w
                                       + h0.x + h1.y + h2.z + h3.w)
                    * (xa0.x + xa1.y + xa2.z + xa3.w + xb0.x + xb1.y + xb2.z + xb3.w)
                    + (float)mn * xbs;
        }
    }
""" + POST
    elif mode == "premul":
        # x пре-домножен: xp[k] = x[k] * 2^-(4*(k%2==...)); нибблы масками.
        # Раскладка кода: байт b = (col_lo, col_hi<<4); uint32 qb[t] = байты
        # 4t..4t+3 = колонки lo 4t..4t+3 и hi 16+4t..16+4t+3.
        # Термы: (q & 0xF)       * x_lo[4t]   * 1
        #        (q & 0xF0)      * x_lo[4t+1] * 2^-4
        #        (q & 0xF00)     * x_lo[4t+2] * 2^-8   ... и т.д.
        #        (q & 0xF0000000)* x_lo[4t+3] ...
        # hi-часть: (q>>4)&0xF... нет -- маски по тем же байтам с >>4 нельзя
        # без сдвига; берём (q & 0xF0) для hi? Нет: hi-ниббл байта --
        # (q & 0xF0) уже покрыт... РАЗДЕЛЕНИЕ: lo-нибблы = маски 0x0F в 4
        # байтах, hi-нибблы = маски 0xF0. Обе группы без сдвигов:
        #   dv += float(q & 0x0000000F) * xl0
        #       + float(q & 0x000000F0) * xh0 * 2^-4
        #       + float(q & 0x00000F00) * xl1 * 2^-8
        #       + float(q & 0x0000F000) * xh1 * 2^-12 ... (чередование lo/hi!)
        # где xl_k -- колонка 4t+k (lo), xh_k -- колонка 16+4t+k (hi).
        dec = ""
        if xbits >= 1:
            dec += "            uint hb = qb[4];\n" + _mk_mult("hb", 4)
        if xbits >= 2:
            dec += "            uint hb2 = qb[5];\n" + _mk_mult("hb2", 5)
        if xbits == 0:
            terms = []
            for t in range(4):
                for byte in range(4):
                    lo_col = 4*t + byte          # x-индекс lo
                    hi_col = 16 + 4*t + byte     # hi
                    mlo = 0xF << (8*byte)
                    mhi = 0xF0 << (8*byte)
                    terms.append(f"                 + (float)(qb{t} & 0x{mlo:08X}u) * xp[{lo_col}] ")
                    terms.append(f"                 + (float)(qb{t} & 0x{mhi:08X}u) * xph[{hi_col-16}]")
            body_terms = "\n".join(terms)
            body = PRE + """
    for (uint p = lane; p < NB; p += 32) {
        float xp[16];
        float xph[16];
        {
            device const float* xf = (device const float*)(x4) + p*32;
            for (uint k = 0; k < 16; k++) {
                float sc = exp2(-8.0f * (float)(k % 4));   // 2^-8 на байт
                xp[k]  = xf[k]      * sc * ((k % 4) ? 1.0f : 1.0f);
                xph[k] = xf[16 + k] * sc * 0.0625f;         // ещё 2^-4 для hi
            }
        }
        float xbs = xbsum[p];
        for (uint j = 0; j < RS; j++) {
""" + QLOADS + """
            uint qb0 = qb[0], qb1 = qb[1], qb2 = qb[2], qb3 = qb[3];
            float dv = 0.0f
""" + body_terms + """;
            acc[j] += (float)s * dv + (float)mn * xbs;
        }
    }
""" + POST
        else:
            # int5/int6: нибблы premult + плоскости мульт-трюком поверх
            # (плоскости добавляются как отдельные dot'ы битов с premult x)
            return None
    else:
        raise ValueError(mode)
    kern = mx.fast.metal_kernel(
        name=f"k4probe_{mode}_gw{4+xbits}_{IN}_{OUT}",
        input_names=["x", "qblk", "qsqm", "ddm", "xbsum"],
        output_names=["out"], header=hdr, source=body)
    _cache[key] = kern
    return kern


def _mk_mult(src, shift):
    ls = []
    for i, reg in enumerate(["l0", "l1", "l2", "l3", "h0", "h1", "h2", "h3"]):
        sh = i * 4
        nib = f"({src} >> 28)" if sh == 28 else (
            f"(({src} >> {sh}) & 0xFu)" if sh else f"({src} & 0xFu)")
        ls.append(f"            {reg} |= as_type<uchar4>(({nib} * {M}) << {shift});")
    return "\n".join(ls) + "\n"


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


rng = np.random.default_rng(0)
CASES = [("cmixK4 8192x2048", 8192, 2048, 0),
         ("cmixV4 2048x8192", 2048, 8192, 0),
         ("cmixK6 8192x2048", 8192, 2048, 2)]

# MLX-двойник по форме для точки отсчёта
def mlx_ref(OUT, IN, bits):
    w = mx.array(rng.standard_normal((OUT, IN)).astype(np.float16))
    wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(wq, sc, bi)
    mb = (wq.size*4 + sc.size*2 + bi.size*2) / 1e6
    xh = mx.array(rng.standard_normal((1, IN)).astype(np.float16)); mx.eval(xh)
    def call():
        return mx.quantized_matmul(xh, wq, scales=sc, biases=bi,
                                   transpose=True, group_size=64, bits=bits)
    return call, mb

print("=== кернель-4 руфлайн-проба (остывшая машина) ===")
for name, OUT, IN, xbits in CASES:
    NB, NSB = IN // 32, IN // 256
    qblk = mx.array(rng.integers(0, 256, (OUT, NB*(16+4*xbits)), dtype=np.uint8))
    qsqm = mx.array(rng.integers(0, 256, (OUT, NB*2), dtype=np.uint8))
    ddm = mx.array((rng.random((OUT, NSB*2), dtype=np.float32)*1e-3).astype(np.float16))
    x = mx.array(rng.standard_normal((1, IN)).astype(np.float32))
    xbsum = mx.sum(x.reshape(1, NB, 32), axis=2)
    mx.eval(qblk, qsqm, ddm, x, xbsum)
    mb = qblk.size/1e6 + qsqm.size/1e6 + ddm.size*2/1e6
    NSG, RS = _k3_cfg(IN, OUT, xbits)
    n_tg = OUT // (NSG*RS)

    def call_mode(mode):
        kern = make(IN, OUT, xbits, NSG, RS, mode)
        return kern(inputs=[x, qblk, qsqm, ddm, xbsum],
                    grid=(n_tg*NSG*32, 1, 1), threadgroup=(NSG*32, 1, 1),
                    output_shapes=[(1, OUT)], output_dtypes=[mx.float32])[0]

    def call_full():
        kern = _get_kernel_k3(IN, OUT, xbits, NSG, RS)
        return kern(inputs=[x, qblk, qsqm, ddm, xbsum],
                    grid=(n_tg*NSG*32, 1, 1), threadgroup=(NSG*32, 1, 1),
                    output_shapes=[(1, OUT)], output_dtypes=[mx.float32])[0]

    modes = ["mem", "nodot"] + (["premul"] if xbits == 0 else [])
    mlx_call, mlx_mb = mlx_ref(OUT, IN, 4 + xbits)

    # premul-корректность vs full (не бит-в-бит, allclose)
    if xbits == 0:
        a = np.array(call_full()); b = np.array(call_mode("premul"))
        rel = float(np.max(np.abs(a-b)) / (np.abs(a).max() + 1e-6))
        if rel > 1e-4:
            print(f"  !! premul rel diff {rel:.1e}")

    res = {"full": [], "mlx": []}
    for m in modes: res[m] = []
    for _ in range(3):
        res["full"] += bench(call_full)
        for m in modes: res[m] += bench(lambda m=m: call_mode(m))
        res["mlx"] += bench(mlx_call)
    fu = float(np.median(res["full"]))
    line = f"{name:16s} full={fu:6.3f} [{mb/fu:5.1f}]"
    for m in modes:
        v = float(np.median(res[m]))
        line += f"  {m}={v:6.3f} [{mb/v:5.1f}]"
    v = float(np.median(res["mlx"]))
    line += f"  MLX={v:6.3f} [{mlx_mb/v:5.1f}]"
    print(line, flush=True)
