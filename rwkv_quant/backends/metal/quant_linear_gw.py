"""GEMV/GEMM для формата v2 (gw_mode="sb6"): блок 32, суперблок 256,
scale/min блока = 6-битные qs/qm против fp16-пары d/dm суперблока.

Раскладка кодов -- блок-локальный split (schema.pack_nib_block): блок из 32
колонок = 16 байт = ОДИН uint4-лоад; lo-нибблы = колонки 0..15 блока,
hi = 16..31. Для bits=5 -- битплоскость qh (schema.pack_bitplane): бит c
строки = 5-й бит кода колонки c, блок = один uint32-лоад. Для bits=6 --
ВТОРАЯ битплоскость qh2 тем же механизмом (6-й бит), независимая от qh.

Математика на блок b (числа как в кернеле):
    s = (half)(qs[b] * (float)d[b/8])      -- бит-в-бит с writer/reader
    m = (half)(qm[b] * (float)dm[b/8])
    acc += s * dot(x_b, q_b) + m * xbsum[b]
xbsum[n, b] = sum(x[n, b*32 : b*32+32]) предвычисляется снаружи (аналог
xsum в v2): min-поправки на блок нельзя вынести одной строкой, как -8*sum(x)
per-row у biased v1-раскладки.

Скелет threadgroup'а -- как в quant_linear_v2 packed: R строк на группу из
TG потоков, страйд по блокам, simd_sum-редукция."""
import numpy as np
import torch
import mlx.core as mx

from ...formats.schema import unpack6

_gw_kernel_cache = {}

TG = 32
R = 8
GEMM_MIN_BATCH = 16


def _get_kernel_gw(IN: int, OUT: int, xbits: int, out_per: int = 0):
    # out_per > 0: мульти-вход (фьюз r/k/v) -- x это стек [OUT/out_per, IN],
    # строка row берёт вход номер row/out_per; xbsum стекован так же.
    # Математика каждой строки бит-в-бит с одиночным кернелем.
    # xbits: 0 = int4 (только нибблы), 1 = int5 (+qh, бит4), 2 = int6
    # (+qh +qh2, биты 4 и 5).
    assert xbits in (0, 1, 2)
    key = (IN, OUT, xbits, out_per)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0, "sb6-кернель: IN кратен суперблоку 256"
    if out_per:
        assert OUT % out_per == 0 and out_per % R == 0
    NB, NSB = IN // 32, IN // 256

    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint TG    = {TG};
constant uint R     = {R};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
constant uint OUT_PER = {out_per if out_per else OUT};
"""
    guard_hot  = "" if OUT % R == 0 else "            if (row0 + j >= OUT_C) break;\n"
    guard_tail = "" if OUT % R == 0 else "        if (row >= OUT_C) break;\n"

    qh_body = """
            uint hb = ((device const uint*)(qh + (row0+j)*(IN_C/8)))[p];
            l0 |= uchar4((uint4(hb) >> uint4( 0, 1, 2, 3)) & 1u) << 4;
            l1 |= uchar4((uint4(hb) >> uint4( 4, 5, 6, 7)) & 1u) << 4;
            l2 |= uchar4((uint4(hb) >> uint4( 8, 9,10,11)) & 1u) << 4;
            l3 |= uchar4((uint4(hb) >> uint4(12,13,14,15)) & 1u) << 4;
            h0 |= uchar4((uint4(hb) >> uint4(16,17,18,19)) & 1u) << 4;
            h1 |= uchar4((uint4(hb) >> uint4(20,21,22,23)) & 1u) << 4;
            h2 |= uchar4((uint4(hb) >> uint4(24,25,26,27)) & 1u) << 4;
            h3 |= uchar4((uint4(hb) >> uint4(28,29,30,31)) & 1u) << 4;
""" if xbits >= 1 else ""

    qh2_body = """
            uint hb2 = ((device const uint*)(qh2 + (row0+j)*(IN_C/8)))[p];
            l0 |= uchar4((uint4(hb2) >> uint4( 0, 1, 2, 3)) & 1u) << 5;
            l1 |= uchar4((uint4(hb2) >> uint4( 4, 5, 6, 7)) & 1u) << 5;
            l2 |= uchar4((uint4(hb2) >> uint4( 8, 9,10,11)) & 1u) << 5;
            l3 |= uchar4((uint4(hb2) >> uint4(12,13,14,15)) & 1u) << 5;
            h0 |= uchar4((uint4(hb2) >> uint4(16,17,18,19)) & 1u) << 5;
            h1 |= uchar4((uint4(hb2) >> uint4(20,21,22,23)) & 1u) << 5;
            h2 |= uchar4((uint4(hb2) >> uint4(24,25,26,27)) & 1u) << 5;
            h3 |= uchar4((uint4(hb2) >> uint4(28,29,30,31)) & 1u) << 5;
""" if xbits >= 2 else ""

    body = """
    uint g    = threadgroup_position_in_grid.x;
    uint n    = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint row0 = g * R;

    uint xi = (n * (OUT_C / OUT_PER)) + row0 / OUT_PER;
    device const float4* x4 = (device const float4*)(x + xi*IN_C);
    float acc[R];
    for (uint j = 0; j < R; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += TG) {          // p -- блок из 32 колонок
        float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
        float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
        float xbs;
        if (XIN) {
            const float4 F1 = float4(1.0f);
            xbs = dot(xa0, F1) + dot(xa1, F1) + dot(xa2, F1) + dot(xa3, F1)
                + dot(xb0, F1) + dot(xb1, F1) + dot(xb2, F1) + dot(xb3, F1);
        } else {
            xbs = xbsum[xi*NB + p];
        }
        for (uint j = 0; j < R; j++) {
GUARD_HOT            uint4 qw = ((device const uint4*)(codes + (row0+j)*(IN_C/2)))[p];
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
    for (uint j = 0; j < R; j++) {
        uint row = row0 + j;
GUARD_TAIL        float a = simd_sum(acc[j]);
        if (lane == 0)
            out[n*OUT_C + row] = a;
    }
"""
    body = body.replace("GUARD_HOT", guard_hot).replace("GUARD_TAIL", guard_tail)
    kern = mx.fast.metal_kernel(
        name=f"quant_linear_gw{4 + xbits}_{IN}_{OUT}",
        input_names=["x", "codes", "qs", "qm", "d", "dm", "qh", "qh2", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern



RB_NB = 4  # дефолт RB для N-батчевого кернеля
NB_CHUNK = 4          # оптимум свипа 19.07: N=4 = 2.28 мс/кол (spill дальше)
_RB_FOR_NN = {2: 4, 3: 2, 4: 2}  # bench_gw_nb_sweep.py
GEMM_MIN_BATCH_NB = 128  # ниже -- чанки по NB_CHUNK, выше -- dequant-GEMM
NB_V2 = False         # nb2: загрузки подняты (x один раз на (p,n)), декод дублируется по n


def _get_kernel_gw_nb(IN: int, OUT: int, xbits: int, NN: int, rb: int = 0):
    """N-батчевый GEMV (2 <= NN < GEMM_MIN_BATCH): веса блока декодируются
    ОДИН раз и применяются ко всем NN колонкам x (x мал и кэшируется).
    Математика на пару (строка, колонка) бит-в-бит с _get_kernel_gw:
    тот же порядок dot'ов, тот же simd_sum. Устраняет N-кратное
    перечитывание весов из DRAM (диагноз сессии 19.07-12: T=4 стоил
    2.18x T=1, веса ехали на каждую колонку заново)."""
    rb = rb or RB_NB
    assert xbits in (0, 1, 2) and NN >= 2
    key = ("nb", IN, OUT, xbits, NN, rb)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0
    NB, NSB = IN // 32, IN // 256

    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint TG    = {TG};
constant uint RB    = {rb};
constant uint NN    = {NN};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
"""
    guard_hot  = "" if OUT % rb == 0 else "            if (row0 + j >= OUT_C) break;\n"
    guard_tail = "" if OUT % rb == 0 else "        if (row0 + j >= OUT_C) break;\n"

    qh_body = """
            uint hb = ((device const uint*)(qh + (row0+j)*(IN_C/8)))[p];
            l0 |= uchar4((uint4(hb) >> uint4( 0, 1, 2, 3)) & 1u) << 4;
            l1 |= uchar4((uint4(hb) >> uint4( 4, 5, 6, 7)) & 1u) << 4;
            l2 |= uchar4((uint4(hb) >> uint4( 8, 9,10,11)) & 1u) << 4;
            l3 |= uchar4((uint4(hb) >> uint4(12,13,14,15)) & 1u) << 4;
            h0 |= uchar4((uint4(hb) >> uint4(16,17,18,19)) & 1u) << 4;
            h1 |= uchar4((uint4(hb) >> uint4(20,21,22,23)) & 1u) << 4;
            h2 |= uchar4((uint4(hb) >> uint4(24,25,26,27)) & 1u) << 4;
            h3 |= uchar4((uint4(hb) >> uint4(28,29,30,31)) & 1u) << 4;
""" if xbits >= 1 else ""

    qh2_body = """
            uint hb2 = ((device const uint*)(qh2 + (row0+j)*(IN_C/8)))[p];
            l0 |= uchar4((uint4(hb2) >> uint4( 0, 1, 2, 3)) & 1u) << 5;
            l1 |= uchar4((uint4(hb2) >> uint4( 4, 5, 6, 7)) & 1u) << 5;
            l2 |= uchar4((uint4(hb2) >> uint4( 8, 9,10,11)) & 1u) << 5;
            l3 |= uchar4((uint4(hb2) >> uint4(12,13,14,15)) & 1u) << 5;
            h0 |= uchar4((uint4(hb2) >> uint4(16,17,18,19)) & 1u) << 5;
            h1 |= uchar4((uint4(hb2) >> uint4(20,21,22,23)) & 1u) << 5;
            h2 |= uchar4((uint4(hb2) >> uint4(24,25,26,27)) & 1u) << 5;
            h3 |= uchar4((uint4(hb2) >> uint4(28,29,30,31)) & 1u) << 5;
""" if xbits >= 2 else ""

    body = """
    uint g    = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint row0 = g * RB;

    float acc[RB * NN];
    for (uint j = 0; j < RB * NN; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += TG) {
        for (uint j = 0; j < RB; j++) {
GUARD_HOT            uint4 qw = ((device const uint4*)(codes + (row0+j)*(IN_C/2)))[p];
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
""" + qh_body + qh2_body + """
            float4 w0 = float4(l0.x, l0.y, l0.z, l0.w);
            float4 w1 = float4(l1.x, l1.y, l1.z, l1.w);
            float4 w2 = float4(l2.x, l2.y, l2.z, l2.w);
            float4 w3 = float4(l3.x, l3.y, l3.z, l3.w);
            float4 w4 = float4(h0.x, h0.y, h0.z, h0.w);
            float4 w5 = float4(h1.x, h1.y, h1.z, h1.w);
            float4 w6 = float4(h2.x, h2.y, h2.z, h2.w);
            float4 w7 = float4(h3.x, h3.y, h3.z, h3.w);
            uint sbi = (row0+j)*NSB + p/8;
            half  s  = (half)((float)qs[(row0+j)*NB + p] * (float)d[sbi]);
            half  mn = (half)((float)qm[(row0+j)*NB + p] * (float)dm[sbi]);
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
                acc[j*NN + n] += (float)s * dv + (float)mn * xbsA[n];
            }
        }
    }
    for (uint j = 0; j < RB; j++) {
GUARD_TAIL        for (uint n = 0; n < NN; n++) {
            float a = simd_sum(acc[j*NN + n]);
            if (lane == 0)
                out[n*OUT_C + row0 + j] = a;
        }
    }
"""
    body = body.replace("GUARD_HOT", guard_hot).replace("GUARD_TAIL", guard_tail)
    kern = mx.fast.metal_kernel(
        name=f"quant_linear_gw{4 + xbits}_nb{NN}r{rb}_{IN}_{OUT}",
        input_names=["x", "codes", "qs", "qm", "d", "dm", "qh", "qh2", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern



def _get_kernel_gw_nb2(IN: int, OUT: int, xbits: int, NN: int, rb: int = 2):
    """nb-вариант 2: все ЗАГРУЗКИ подняты из внутренних циклов.
    Полный проход блока p: qw/hb/hb2/s/mn для RB строк грузятся один раз,
    x-блок (8 float4) грузится один раз на (p, n) и переиспользуется
    всеми RB строками; декод весов повторяется по n (чистый ALU, есть
    запас: ALU ~32% при memory-bound). x-трафик: NN вместо RB*NN
    (диагноз bench_gw_nb_groups: cmix N=4 = 3.4x N=1 из-за x-reload).
    Математика бит-в-бит: тот же порядок 8 dot'ов и acc."""
    assert xbits in (0, 1, 2) and NN >= 2
    key = ("nb2", IN, OUT, xbits, NN, rb)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0
    NB, NSB = IN // 32, IN // 256

    hdr = f"""
constant uint IN_C  = {IN};
constant uint OUT_C = {OUT};
constant uint TG    = {TG};
constant uint RB    = {rb};
constant uint NN    = {NN};
constant uint NB    = {NB};
constant uint NSB   = {NSB};
"""
    guard_hot  = "" if OUT % rb == 0 else "            if (row0 + j >= OUT_C) break;\n"
    guard_tail = "" if OUT % rb == 0 else "        if (row0 + j >= OUT_C) break;\n"

    decl_h  = "        uint hbA[RB];\n"  if xbits >= 1 else ""
    decl_h2 = "        uint hb2A[RB];\n" if xbits >= 2 else ""
    load_h  = "            hbA[j] = ((device const uint*)(qh + (row0+j)*(IN_C/8)))[p];\n" if xbits >= 1 else ""
    load_h2 = "            hb2A[j] = ((device const uint*)(qh2 + (row0+j)*(IN_C/8)))[p];\n" if xbits >= 2 else ""

    dec_h = """
                uint hb = hbA[j];
                l0 |= uchar4((uint4(hb) >> uint4( 0, 1, 2, 3)) & 1u) << 4;
                l1 |= uchar4((uint4(hb) >> uint4( 4, 5, 6, 7)) & 1u) << 4;
                l2 |= uchar4((uint4(hb) >> uint4( 8, 9,10,11)) & 1u) << 4;
                l3 |= uchar4((uint4(hb) >> uint4(12,13,14,15)) & 1u) << 4;
                h0 |= uchar4((uint4(hb) >> uint4(16,17,18,19)) & 1u) << 4;
                h1 |= uchar4((uint4(hb) >> uint4(20,21,22,23)) & 1u) << 4;
                h2 |= uchar4((uint4(hb) >> uint4(24,25,26,27)) & 1u) << 4;
                h3 |= uchar4((uint4(hb) >> uint4(28,29,30,31)) & 1u) << 4;
""" if xbits >= 1 else ""
    dec_h2 = """
                uint hb2 = hb2A[j];
                l0 |= uchar4((uint4(hb2) >> uint4( 0, 1, 2, 3)) & 1u) << 5;
                l1 |= uchar4((uint4(hb2) >> uint4( 4, 5, 6, 7)) & 1u) << 5;
                l2 |= uchar4((uint4(hb2) >> uint4( 8, 9,10,11)) & 1u) << 5;
                l3 |= uchar4((uint4(hb2) >> uint4(12,13,14,15)) & 1u) << 5;
                h0 |= uchar4((uint4(hb2) >> uint4(16,17,18,19)) & 1u) << 5;
                h1 |= uchar4((uint4(hb2) >> uint4(20,21,22,23)) & 1u) << 5;
                h2 |= uchar4((uint4(hb2) >> uint4(24,25,26,27)) & 1u) << 5;
                h3 |= uchar4((uint4(hb2) >> uint4(28,29,30,31)) & 1u) << 5;
""" if xbits >= 2 else ""

    body = """
    uint g    = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint row0 = g * RB;

    float acc[RB * NN];
    for (uint j = 0; j < RB * NN; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += TG) {
        uint4 qwA[RB];
        half  sA[RB], mnA[RB];
DECL_HDECL_H2        float xbsA[NN];
        for (uint j = 0; j < RB; j++) {
GUARD_HOT            qwA[j] = ((device const uint4*)(codes + (row0+j)*(IN_C/2)))[p];
LOAD_HLOAD_H2            uint sbi = (row0+j)*NSB + p/8;
            sA[j]  = (half)((float)qs[(row0+j)*NB + p] * (float)d[sbi]);
            mnA[j] = (half)((float)qm[(row0+j)*NB + p] * (float)dm[sbi]);
        }
        for (uint n = 0; n < NN; n++) xbsA[n] = xbsum[n*NB + p];

        for (uint n = 0; n < NN; n++) {
            device const float4* x4 = (device const float4*)(x + n*IN_C);
            float4 xa0 = x4[p*8+0], xa1 = x4[p*8+1], xa2 = x4[p*8+2], xa3 = x4[p*8+3];
            float4 xb0 = x4[p*8+4], xb1 = x4[p*8+5], xb2 = x4[p*8+6], xb3 = x4[p*8+7];
            for (uint j = 0; j < RB; j++) {
GUARD_HOT2                uint4 qw = qwA[j];
                uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
                uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
                uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
                uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
                uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
                uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
                uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
                uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
DEC_HDEC_H2                float dv = dot(xa0, float4(l0.x, l0.y, l0.z, l0.w))
                         + dot(xa1, float4(l1.x, l1.y, l1.z, l1.w))
                         + dot(xa2, float4(l2.x, l2.y, l2.z, l2.w))
                         + dot(xa3, float4(l3.x, l3.y, l3.z, l3.w))
                         + dot(xb0, float4(h0.x, h0.y, h0.z, h0.w))
                         + dot(xb1, float4(h1.x, h1.y, h1.z, h1.w))
                         + dot(xb2, float4(h2.x, h2.y, h2.z, h2.w))
                         + dot(xb3, float4(h3.x, h3.y, h3.z, h3.w));
                acc[j*NN + n] += (float)sA[j] * dv + (float)mnA[j] * xbsA[n];
            }
        }
    }
    for (uint j = 0; j < RB; j++) {
GUARD_TAIL        for (uint n = 0; n < NN; n++) {
            float a = simd_sum(acc[j*NN + n]);
            if (lane == 0)
                out[n*OUT_C + row0 + j] = a;
        }
    }
"""
    body = (body.replace("GUARD_HOT2", "" if OUT % rb == 0 else "                if (row0 + j >= OUT_C) break;\n")
                .replace("GUARD_HOT", guard_hot).replace("GUARD_TAIL", guard_tail)
                .replace("DECL_H2", decl_h2).replace("DECL_H", decl_h)
                .replace("LOAD_H2", load_h2).replace("LOAD_H", load_h)
                .replace("DEC_H2", dec_h2).replace("DEC_H", dec_h))
    kern = mx.fast.metal_kernel(
        name=f"quant_linear_gw{4 + xbits}_nb2_{NN}r{rb}_{IN}_{OUT}",
        input_names=["x", "codes", "qs", "qm", "d", "dm", "qh", "qh2", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern




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
_XB_DUMMY = None
K3_XSUM = False   # xbsum в кернеле: НЕ бит-в-бит (порядок сумм), включать после ppl-гейта


def _xb_dummy():
    global _XB_DUMMY
    if _XB_DUMMY is None:
        _XB_DUMMY = mx.zeros((1,), dtype=mx.float32)
    return _XB_DUMMY


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
    return "\n".join(ls) + "\n"


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


def _get_kernel_k3(IN, OUT, xbits, NSG, RS, out_per=0, xin=False):
    """N=1 GEMV (out_per=0) либо фьюз r/k/v (out_per>0, вход-стек)."""
    assert xbits in (0, 1, 2)
    key = ("k3", IN, OUT, xbits, NSG, RS, out_per, xin)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    op = out_per if out_per else OUT
    assert op % (NSG * RS) == 0
    hdr = _k3_hdr(IN, OUT, xbits, NSG, RS,
                  f"constant uint OUT_PER = {op};\nconstant bool XIN = {str(bool(xin)).lower()};")
    dec = _K3_DECODE
    if xbits >= 1:
        dec += "            uint hb = qb[4];\n" + _k3_plane("hb", 4)
    if xbits >= 2:
        dec += "            uint hb2 = qb[5];\n" + _k3_plane("hb2", 5)
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
        name=f"k3_gw{4 + xbits}_s{NSG}r{RS}_{IN}_{OUT}_{op}{'x' if xin else ''}",
        input_names=["x", "qblk", "qsqm", "ddm", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern


def _get_kernel_k3nb(IN, OUT, xbits, NN, NSG, RS, xin=False):
    """N-батчевый GEMV кернеля-3: веса блока декодируются один раз на NN
    колонок (структура _get_kernel_gw_nb, раскладка кернеля-3)."""
    assert xbits in (0, 1, 2) and NN >= 2
    key = ("k3nb", IN, OUT, xbits, NN, NSG, RS, xin)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    hdr = _k3_hdr(IN, OUT, xbits, NSG, RS,
                  f"constant uint NN = {NN};\nconstant bool XIN = {str(bool(xin)).lower()};")
    dec = _K3_DECODE
    if xbits >= 1:
        dec += "            uint hb = qb[4];\n" + _k3_plane("hb", 4)
    if xbits >= 2:
        dec += "            uint hb2 = qb[5];\n" + _k3_plane("hb2", 5)
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
        float xbsA[NN];
        if (XIN) {
            const float4 F1 = float4(1.0f);
            for (uint n = 0; n < NN; n++) {
                device const float4* x4s = (device const float4*)(x + n*IN_C);
                xbsA[n] = dot(x4s[p*8+0], F1) + dot(x4s[p*8+1], F1)
                        + dot(x4s[p*8+2], F1) + dot(x4s[p*8+3], F1)
                        + dot(x4s[p*8+4], F1) + dot(x4s[p*8+5], F1)
                        + dot(x4s[p*8+6], F1) + dot(x4s[p*8+7], F1);
            }
        } else {
            for (uint n = 0; n < NN; n++) xbsA[n] = xbsum[n*NB + p];
        }
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
                acc[j*NN + n] += (float)s * dv + (float)mn * xbsA[n];
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
        name=f"k3nb_gw{4 + xbits}_n{NN}s{NSG}r{RS}_{IN}_{OUT}{'x' if xin else ''}",
        input_names=["x", "qblk", "qsqm", "ddm", "xbsum"],
        output_names=["out"],
        header=hdr, source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern


class GwQuantLinear:
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
        if self._k3 and not NB_V2:
            xin = bool(K3_XSUM)
            xbsum = (_xb_dummy() if xin
                     else mx.sum(x2d.reshape(c, self.NB, 32), axis=2))
            NSG, RS = _k3_cfg_nb(self.in_features, self.out_features,
                                 self.xbits)
            kern = _get_kernel_k3nb(self.in_features, self.out_features,
                                    self.xbits, c, NSG, RS, xin=xin)
            n_tg = self.out_features // (NSG * RS)
            return kern(
                inputs=[x2d, self.qblk, self.qsqm, self.ddm, xbsum],
                grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                output_shapes=[(c, self.out_features)],
                output_dtypes=[mx.float32],
            )[0]
        xbsum = mx.sum(x2d.reshape(c, self.NB, 32), axis=2)
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
        if self._k3:
            xin = bool(K3_XSUM)
            xbsum = (_xb_dummy() if xin
                     else mx.sum(x2d.reshape(1, self.NB, 32), axis=2))
            NSG, RS = _k3_cfg(self.in_features, self.out_features, self.xbits)
            kern = _get_kernel_k3(self.in_features, self.out_features,
                                  self.xbits, NSG, RS, xin=xin)
            n_tg = self.out_features // (NSG * RS)
            return kern(
                inputs=[x2d, self.qblk, self.qsqm, self.ddm, xbsum],
                grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                output_shapes=[(1, self.out_features)],
                output_dtypes=[mx.float32],
            )[0]
        xbsum = mx.sum(x2d.reshape(1, self.NB, 32), axis=2)
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
        if self._k3:
            xin = bool(K3_XSUM)
            xbsum = (_xb_dummy() if xin
                     else mx.sum(xstack.reshape(self.K, self.NB, 32), axis=2))
            NSG, RS = _k3_cfg(self.in_features, self.out_per, self.xbits)
            kern = _get_kernel_k3(self.in_features, self.out_features,
                                  self.xbits, NSG, RS, out_per=self.out_per,
                                  xin=xin)
            n_tg = self.out_features // (NSG * RS)
            out = kern(
                inputs=[xstack, self.qblk, self.qsqm, self.ddm, xbsum],
                grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
                output_shapes=[(1, self.out_features)],
                output_dtypes=[mx.float32],
            )[0]
            return out.reshape(self.K, self.out_per)
        xbsum = mx.sum(xstack.reshape(self.K, self.NB, 32), axis=2)
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
