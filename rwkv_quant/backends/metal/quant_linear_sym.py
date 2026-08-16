"""GEMV для раскладки sym (Q6_K-подобная, gw_mode="sym"): блок 16 БЕЗ min,
scale блока = int8-код против одной fp16 d на суперблок из 16 блоков.

ЭТО УРЕЗАННЫЙ КЕРНЕЛЬ-3, А НЕ НОВАЯ МЕХАНИКА. Отличий от sb6 ровно три,
и каждое УБИРАЕТ работу, а не добавляет:

  1. Нет min. У sb6 на блок приходится `acc += s*dv + m*xbsum`; здесь
     второго слагаемого нет вовсе, а с ним -- и второго квального
     скаляра (qm), и второй fp16-константы суперблока (dm).
  2. Масштаб блока -- ЦЕЛЫЙ байт (int8) против шести бит у sb6, поэтому
     распаковывать его не нужно: `s = (half)((float)qs[b] * (float)d[b/16])`
     читается одним лоадом. Формула та же, что в writer/reader/codec,
     включая half-роундтрип -- иначе кернель разойдётся с файлом на
     последнем бите мантиссы.
  3. Блок вдвое уже (16 против 32), а суперблок тот же (256 весов).

РАСКЛАДКА ЗАГРУЗЧИКА (интерлив, как K3 у sb6): коды и масштабы одного
блока лежат рядом, чтобы блок брался тремя потоками, а не пятью.
Дисковая раскладка при этом не меняется -- см. formats/codec.py, диск
канонический, интерлив есть деталь загрузчика.

  bits=8: qblk[row, b] = 16 байт ЗНАКОВЫХ кодов = ровно один uint4-лоад
          на блок. Никаких битплоскостей: код -- это байт.
  bits=6: коды лежат СО СДВИГОМ +32 в 0..63 (ровно как в block_q6_K у
          llama.cpp), младший ниббл -- блок-локальным split'ом, биты 4 и
          5 -- двумя битплоскостями. На ПАРУ блоков (32 колонки) это
          16 байт нибблов + 4 байта qh + 4 байта qh2 = 6 uint, то есть
          БАЙТ-В-БАЙТ тот же гранул, что у sb6 при xbits=2. Поэтому
          декодер нибблов и мульт-трюк битплоскостей взяты оттуда без
          единой правки -- меняется только порядок регистров (см. ниже)
          и то, что на пару приходится ДВА масштаба вместо одного.

ПОЧЕМУ ПОРЯДОК РЕГИСТРОВ ДРУГОЙ. Split блок-локальный, а блок теперь 16:
байт j пары несёт колонку j (lo) и j+8 (hi) СВОЕГО блока. Значит
колонки идут l0,l1,h0,h1 (первый блок) и l2,l3,h2,h3 (второй), тогда как
при gs=32 они шли l0..l3,h0..h3. Перепутать это местами -- получить
правдоподобный, но неверный результат: величины те же, порядок не тот.

СДВИГ +32 СНИМАЕТСЯ В КЕРНЕЛЕ, а не в памяти: sum((q-32)*x) =
sum(q*x) - 32*sum(x), и вторая сумма -- это тот же xbsum, что уже
считается для sb6, только на блок 16. При bits=8 сдвига нет, и xbsum не
нужен вовсе -- кернель его не читает.
"""
import numpy as np
import mlx.core as mx

from .quant_linear_gw import GEMM_MIN_BATCH_NB, _gw_kernel_cache

# порядок регистров -> порядок колонок внутри ПАРЫ блоков по 16
_PAIR_REGS = ["l0", "l1", "h0", "h1", "l2", "l3", "h2", "h3"]


def _cfg(IN, OUT):
    """(NSG, RS). Свипов под sym ещё не было -- взята конфигурация,
    которую свип protoD выбрал для тех же форм у sb6; после первого
    A/B-замера её надо пересмотреть, а не считать оптимальной."""
    if OUT >= 32768:
        return (4, 4)
    return (4, 4)


def _plane_pair(src, shift):
    """Мульт-трюк битплоскости (ниббл -> 4 байта одним умножением),
    порядок регистров -- парный."""
    out = []
    for i, reg in enumerate(_PAIR_REGS):
        sh = i * 4
        if sh == 0:
            nib = f"({src} & 0xFu)"
        elif sh == 28:
            nib = f"({src} >> 28)"
        else:
            nib = f"(({src} >> {sh}) & 0xFu)"
        out.append(f"            {reg} |= as_type<uchar4>(({nib} * 0x00204081u"
                   f" & 0x01010101u) << {shift});")
    return "\n".join(out) + "\n"


def _hdr(IN, OUT, NSG, RS, NN, OUT_PER=0):
    """OUT_PER -- ФЬЮЗ r/k/v одним параметром, а не второй копией кернеля.

    Строки K матриц лежат конкатенированными, и строка сама говорит, какой
    вход ей нужен: `xi = row0 / OUT_PER`. При OUT_PER = OUT_C это даёт
    xi = 0 и NK = 1, то есть индексация вырождается в прежнюю `x + n*IN_C`
    ровно, без единого лишнего сложения в горячем цикле (константы
    сворачиваются). Отдельного «фьюзнутого кернеля» специально НЕТ:
    закон 23 -- параллельные реализации расходятся ровно тогда, когда
    правку вносят в одну из них.

    Что перегенерация исходника не изменила нефьюзнутый путь -- не
    заявление, а гейт: выходы GEMV заморожены в файл ДО правки и
    сверяются на равенство (tests/test_sym_fuse_parity.py)."""
    return f"""
constant uint IN_C   = {IN};
constant uint OUT_C  = {OUT};
constant uint OUT_PER= {OUT_PER or OUT};
constant uint NK     = {OUT // (OUT_PER or OUT)};
constant uint NB    = {IN // 16};
constant uint NSB   = {IN // 256};
constant uint NPAIR = {IN // 32};
constant uint NSG   = {NSG};
constant uint RS    = {RS};
constant uint NN    = {NN};
"""


def _get_kernel_sym8(IN, OUT, NSG, RS, NN=1, OUT_PER=0):
    """bits=8: один uint4 = один блок из 16 знаковых кодов."""
    key = ("sym8", IN, OUT, NSG, RS, NN, OUT_PER)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    # все RS строк одного sg обязаны смотреть в ОДИН вход: row0 кратен RS,
    # поэтому достаточно, чтобы OUT_PER делился на RS
    assert not OUT_PER or OUT_PER % RS == 0, (OUT_PER, RS)
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint row0 = tgid * (NSG * RS) + sg * RS;
    uint xi   = row0 / OUT_PER;          // при OUT_PER=OUT_C всегда 0

    device const uint*  qu = (device const uint*)qblk;
    device const char*  sc = (device const char*)qs;
    float acc[RS * NN];
    for (uint j = 0; j < RS * NN; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NB; p += 32) {          // p -- блок из 16 колонок
        for (uint j = 0; j < RS; j++) {
            device const uint* qb = qu + ((row0+j)*NB + p) * 4;
            char4 c0 = as_type<char4>(qb[0]);
            char4 c1 = as_type<char4>(qb[1]);
            char4 c2 = as_type<char4>(qb[2]);
            char4 c3 = as_type<char4>(qb[3]);
            float4 q0 = float4(c0.x, c0.y, c0.z, c0.w);
            float4 q1 = float4(c1.x, c1.y, c1.z, c1.w);
            float4 q2 = float4(c2.x, c2.y, c2.z, c2.w);
            float4 q3 = float4(c3.x, c3.y, c3.z, c3.w);
            half s = (half)((float)sc[(row0+j)*NB + p]
                            * (float)d[(row0+j)*NSB + p/16]);
            for (uint n = 0; n < NN; n++) {
                device const float4* x4 =
                    (device const float4*)(x + (n*NK + xi)*IN_C);
                float dv = dot(x4[p*4+0], q0) + dot(x4[p*4+1], q1)
                         + dot(x4[p*4+2], q2) + dot(x4[p*4+3], q3);
                acc[n*RS + j] += (float)s * dv;
            }
        }
    }
    for (uint n = 0; n < NN; n++)
        for (uint j = 0; j < RS; j++) {
            float a = simd_sum(acc[n*RS + j]);
            if (lane == 0)
                out[n*OUT_C + row0 + j] = a;
        }
"""
    kern = mx.fast.metal_kernel(
        name=f"sym8_s{NSG}r{RS}n{NN}p{OUT_PER or OUT}_{IN}_{OUT}",
        input_names=["x", "qblk", "qs", "d"],
        output_names=["out"],
        header=_hdr(IN, OUT, NSG, RS, NN, OUT_PER), source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern


def _get_kernel_sym6(IN, OUT, NSG, RS, NN=1, OUT_PER=0):
    """bits=6: пара блоков за итерацию -- 6 uint, как у sb6 при xbits=2."""
    key = ("sym6", IN, OUT, NSG, RS, NN, OUT_PER)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0 and OUT % (NSG * RS) == 0
    assert not OUT_PER or OUT_PER % RS == 0, (OUT_PER, RS)
    dec = """
            uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
            uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
            uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
            uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
            uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
            uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
            uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
            uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
            uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
            uint hb = qb[4];
""" + _plane_pair("hb", 4) + "            uint hb2 = qb[5];\n" \
        + _plane_pair("hb2", 5)
    body = """
    uint tgid = threadgroup_position_in_grid.x;
    uint tix  = thread_position_in_threadgroup.x;
    uint sg   = tix / 32;
    uint lane = tix % 32;
    uint row0 = tgid * (NSG * RS) + sg * RS;
    uint xi   = row0 / OUT_PER;          // при OUT_PER=OUT_C всегда 0

    device const uint* qu = (device const uint*)qblk;
    device const char* sc = (device const char*)qs;
    float acc[RS * NN];
    for (uint j = 0; j < RS * NN; j++) acc[j] = 0.0f;

    for (uint p = lane; p < NPAIR; p += 32) {   // p -- ПАРА блоков = 32 кол.
        for (uint j = 0; j < RS; j++) {
            device const uint* qb = qu + ((row0+j)*NPAIR + p) * 6;
""" + dec + """
            float4 f0 = float4(l0.x, l0.y, l0.z, l0.w);
            float4 f1 = float4(l1.x, l1.y, l1.z, l1.w);
            float4 f2 = float4(h0.x, h0.y, h0.z, h0.w);
            float4 f3 = float4(h1.x, h1.y, h1.z, h1.w);
            float4 f4 = float4(l2.x, l2.y, l2.z, l2.w);
            float4 f5 = float4(l3.x, l3.y, l3.z, l3.w);
            float4 f6 = float4(h2.x, h2.y, h2.z, h2.w);
            float4 f7 = float4(h3.x, h3.y, h3.z, h3.w);
            // суперблок из 16 блоков = 8 пар, поэтому оба масштаба пары
            // всегда живут под ОДНОЙ d -- лоад один
            float dsb = (float)d[(row0+j)*NSB + p/8];
            half s0 = (half)((float)sc[(row0+j)*NB + 2*p]     * dsb);
            half s1 = (half)((float)sc[(row0+j)*NB + 2*p + 1] * dsb);
            for (uint n = 0; n < NN; n++) {
                uint xn = n*NK + xi;
                device const float4* x4 =
                    (device const float4*)(x + xn*IN_C);
                float dv0 = dot(x4[p*8+0], f0) + dot(x4[p*8+1], f1)
                          + dot(x4[p*8+2], f2) + dot(x4[p*8+3], f3);
                float dv1 = dot(x4[p*8+4], f4) + dot(x4[p*8+5], f5)
                          + dot(x4[p*8+6], f6) + dot(x4[p*8+7], f7);
                // снятие сдвига +32: sum((q-32)*x) = sum(q*x) - 32*sum(x)
                float b0 = xbsum[xn*NB + 2*p], b1 = xbsum[xn*NB + 2*p + 1];
                acc[n*RS + j] += (float)s0 * (dv0 - 32.0f * b0)
                               + (float)s1 * (dv1 - 32.0f * b1);
            }
        }
    }
    for (uint n = 0; n < NN; n++)
        for (uint j = 0; j < RS; j++) {
            float a = simd_sum(acc[n*RS + j]);
            if (lane == 0)
                out[n*OUT_C + row0 + j] = a;
        }
"""
    kern = mx.fast.metal_kernel(
        name=f"sym6_s{NSG}r{RS}n{NN}p{OUT_PER or OUT}_{IN}_{OUT}",
        input_names=["x", "qblk", "qs", "d", "xbsum"],
        output_names=["out"],
        header=_hdr(IN, OUT, NSG, RS, NN, OUT_PER), source=body,
    )
    _gw_kernel_cache[key] = kern
    return kern


def _dq_writes(regs, scale, base, T="half"):
    """Восемь uchar4 -> 32 half по колонкам пары. ПОРЯДОК КОЛОНОК берётся
    из того же списка _PAIR_REGS, что и в GEMV: там он задан порядком
    dot-произведений с x4[p*8+i], здесь -- смещением записи. Перепутать
    его местами -- получить правдоподобный, но неверный результат
    (величины те же, порядок не тот), и ловится это только сверкой с
    эталоном."""
    out = []
    for i, reg in enumerate(regs):
        for c, comp in enumerate("xyzw"):
            if T == "half":
                # ТЕКСТ ЭТОЙ ВЕТКИ НЕ ТРОГАТЬ: под ним стоит гейт равенства
                # (test_sym_dequant_kernel), а (q-32) в half и в float дают
                # разный последний бит.
                out.append(f"    o[{base + i*4 + c}] = ((half)(float){reg}.{comp}"
                           f" - (half)32.0h) * {scale};")
            else:
                # fp32-выход: вся арифметика во float, масштаб по-прежнему
                # округляется через half -- ровно как codec.dequant_sym,
                # поэтому здесь законно требовать бит-в-бит с ним.
                out.append(f"    o[{base + i*4 + c}] = ((float){reg}.{comp}"
                           f" - 32.0f) * (float){scale};")
    return "\n".join(out)


def _get_kernel_dequant(IN, OUT, bits, T="half"):
    """sym -> плотная матрица ОДНИМ кернелем. T -- "half" или "float".

    ДВА ВЫХОДА, И ОНИ НЕ ВЗАИМОЗАМЕНЯЕМЫ. fp16 -- путь ИНФЕРЕНСА: под ним
    измерены все числа пресета и стоит гейт равенства против прежней
    цепочки MLX-операций, поэтому его текст менять нельзя. fp32 нужен
    QLoRA-базе в rwkv-metal: там веса восстанавливаются на каждый forward
    и обязаны совпадать с нормативным `codec.dequant_sym` бит-в-бит,
    иначе квантованная база добавит свой источник шума поверх калибровки.
    Разница ровно в том, в какой точности берётся `(q-32)*s`; масштаб
    округляется через half в обеих ветках, как в файле.

    ЗАЧЕМ. Прежний `_dequant_w` собирал коды ЦЕПОЧКОЙ MLX-операций
    (concatenate нибблов, две битплоскости через сдвиги), и каждая
    рождала полноразмерный промежуточный тензор. Замер
    (`tests/probe_gemm_split.py`, N=512): деквант 10.68 мс против 7.46 у
    самого матмула, то есть **60% времени GEMM-префилла уходило не на
    матмул**. По байтам ему положено прочитать 13.8 МБ и записать 33.6,
    то есть около 0.5 мс на полосе 90 ГБ/с -- цепочка шла со скоростью
    4.4 ГБ/с.

    Отсюда же следует, ЧЕГО делать НЕ надо: наш матмул по готовой
    плотной матрице (7.46 мс) идёт почти вровень с нативным квантованным
    (6.78), значит тайловый GEMM с декодом в threadgroup-памяти -- это
    следующий шаг, а не первый.

    Декодер нибблов и мульт-трюк битплоскостей взяты из GEMV-кернеля
    ТЕМ ЖЕ генератором строк (`dec`, `_plane_pair`), а не переписаны:
    закон 23 -- параллельные реализации расходятся ровно тогда, когда
    правку вносят в одну из них."""
    key = ("symdq", IN, OUT, bits, T)
    if key in _gw_kernel_cache:
        return _gw_kernel_cache[key]
    assert IN % 256 == 0
    hdr = _hdr(IN, OUT, 1, 1, 1)
    if bits == 8:
        assert (OUT * (IN // 16)) % 256 == 0
        head = """
    uint gid = thread_position_in_grid.x;
    uint row = gid / NB;
    uint b   = gid % NB;
    device const char* c  = (device const char*)qblk;
    device const char* sc = (device const char*)qs;
    half s = (half)((float)sc[row*NB + b] * (float)d[row*NSB + b/16]);
"""
        if T == "half":
            body = head + """    device half* o = out + row*IN_C + b*16;
    for (uint i = 0; i < 16; i++)
        o[i] = (half)((float)c[row*IN_C + b*16 + i]) * s;
"""
        else:
            body = head + """    device float* o = out + row*IN_C + b*16;
    for (uint i = 0; i < 16; i++)
        o[i] = (float)c[row*IN_C + b*16 + i] * (float)s;
"""
        names = ["qblk", "qs", "d"]
    else:
        assert (OUT * (IN // 32)) % 256 == 0
        dec = """
    uint4 qw = uint4(qb[0], qb[1], qb[2], qb[3]);
    uchar4 l0 = as_type<uchar4>(qw.x & 0x0F0F0F0Fu);
    uchar4 l1 = as_type<uchar4>(qw.y & 0x0F0F0F0Fu);
    uchar4 h0 = as_type<uchar4>((qw.x >> 4) & 0x0F0F0F0Fu);
    uchar4 h1 = as_type<uchar4>((qw.y >> 4) & 0x0F0F0F0Fu);
    uchar4 l2 = as_type<uchar4>(qw.z & 0x0F0F0F0Fu);
    uchar4 l3 = as_type<uchar4>(qw.w & 0x0F0F0F0Fu);
    uchar4 h2 = as_type<uchar4>((qw.z >> 4) & 0x0F0F0F0Fu);
    uchar4 h3 = as_type<uchar4>((qw.w >> 4) & 0x0F0F0F0Fu);
    uint hb = qb[4];
""" + _plane_pair("hb", 4) + "    uint hb2 = qb[5];\n" + _plane_pair("hb2", 5)
        body = """
    uint gid = thread_position_in_grid.x;
    uint row = gid / NPAIR;
    uint p   = gid % NPAIR;
    device const uint* qb = ((device const uint*)qblk) + gid*6;
""" + dec + """
    device const char* sc = (device const char*)qs;
    float dsb = (float)d[row*NSB + p/8];
    half s0 = (half)((float)sc[row*NB + 2*p]     * dsb);
    half s1 = (half)((float)sc[row*NB + 2*p + 1] * dsb);
    device """ + T + """* o = out + row*IN_C + p*32;
""" + _dq_writes(_PAIR_REGS[:4], "s0", 0, T) + "\n" \
   + _dq_writes(_PAIR_REGS[4:], "s1", 16, T) + "\n"
        names = ["qblk", "qs", "d"]
    kern = mx.fast.metal_kernel(
        name=f"symdq{bits}_{IN}_{OUT}_{T}", input_names=names,
        output_names=["out"], header=hdr, source=body)
    _gw_kernel_cache[key] = kern
    return kern


NB_MAX = 4          # столько колонок за один launch в N-батчевом режиме

# вернуть прежний деквант цепочкой MLX-операций (эталон гейта и «до» в A/B)
DEQUANT_REF = __import__("os").environ.get("RWKVQ_DQ_REF") == "1"


class SymQuantLinear:
    """Linear по sym-тензору. Интерфейс как у GwQuantLinear:
    __call__(x [..., IN]) -> [..., OUT] fp32.

    Хранит ТОЛЬКО интерлив (память 1x): qblk + qs + d. Дисковые буферы
    после конструктора не нужны никому.
    """

    def __init__(self, qt):
        assert qt.gw_mode == "sym", qt.gw_mode
        assert qt.gw_gs == 16 and qt.gw_sb == 16, (qt.gw_gs, qt.gw_sb)

        def npy(x):
            return None if x is None else x.numpy()

        self._build(shape=qt.shape, bits=qt.bits, codes=npy(qt.codes),
                    codes_packed=npy(qt.codes_packed), qh=npy(qt.gw_qh),
                    qh2=npy(qt.gw_qh2), qs=qt.gw_qs.numpy(),
                    d=qt.gw_d.numpy())

    @classmethod
    def from_buffers(cls, *, shape, bits, qs, d, codes=None,
                     codes_packed=None, qh=None, qh2=None):
        """То же самое из NUMPY-буферов, БЕЗ torch и без QuantizedTensor.

        Нужен потребителям формата (rwkv-metal, QLoRA-база), которые
        читают файл через `codec.open_rwkvq` и торча не имеют вовсе.
        Отдельной реализации интерлива у них быть не должно: закон 23 --
        параллельные реализации расходятся ровно тогда, когда правку
        вносят в одну из них."""
        obj = cls.__new__(cls)
        obj._build(shape=shape, bits=bits, codes=codes,
                   codes_packed=codes_packed, qh=qh, qh2=qh2, qs=qs, d=d)
        return obj

    def _build(self, *, shape, bits, codes, codes_packed, qh, qh2, qs, d):
        OUT, IN = shape
        assert IN % 256 == 0, f"sym-кернель: IN={IN} не кратен суперблоку 256"
        self.out_features, self.in_features = OUT, IN
        self.bits = bits
        self.NB, self.NSB = IN // 16, IN // 256

        if bits == 8:
            assert codes is not None, "sym@8 хранит коды в codes"
            blk = np.asarray(codes).view(np.uint8)          # знаковые байты
        else:
            assert bits == 6 and codes_packed is not None
            NP = IN // 32
            blk = np.concatenate(
                [np.asarray(codes_packed).reshape(OUT, NP, 16),
                 np.asarray(qh).reshape(OUT, NP, 4),
                 np.asarray(qh2).reshape(OUT, NP, 4)], axis=2)
        self.qblk = mx.array(np.ascontiguousarray(blk.reshape(OUT, -1)))
        self.qs = mx.array(np.ascontiguousarray(
            np.asarray(qs).view(np.uint8)))                 # int8 as bytes
        self.d = mx.array(np.ascontiguousarray(np.asarray(d)))
        mx.eval(self.qblk, self.qs, self.d)
        # (NSG, RS) в обход _cfg -- только для свипа. В проде None: конфиг
        # обязан выбираться таблицей, а не тем, что кто-то забыл сбросить.
        self.cfg_override = None

    def _dequant_w(self, dtype=mx.float16):
        """sym -> плотная [OUT, IN] ОДНИМ кернелем (транзиент на вызов).

        dtype=fp16 -- умолчание и путь инференса, под ним измерен пресет.
        dtype=fp32 -- для QLoRA-базы: бит-в-бит с `codec.dequant_sym`.

        Прежняя реализация цепочкой MLX-операций осталась как
        `_dequant_w_ref` и служит эталоном гейта
        (tests/test_sym_dequant_kernel.py, требуется РАВЕНСТВО)."""
        if DEQUANT_REF and dtype == mx.float16:
            return self._dequant_w_ref()   # A/B подменой ОДНОГО флага
        OUT, IN = self.out_features, self.in_features
        T = "half" if dtype == mx.float16 else "float"
        kern = _get_kernel_dequant(IN, OUT, self.bits, T)
        n = OUT * (IN // (16 if self.bits == 8 else 32))
        return kern(
            inputs=[self.qblk, self.qs, self.d],
            grid=(n, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(OUT, IN)], output_dtypes=[dtype],
        )[0]

    def _dequant_w_ref(self):
        """ЭТАЛОН: тот же деквант цепочкой MLX-операций. Медленный (60%
        времени GEMM-префилла), но не зависящий от кернеля -- ради этого
        и оставлен."""
        OUT, IN = self.out_features, self.in_features
        if self.bits == 8:
            q = mx.view(self.qblk, mx.int8).reshape(OUT, IN).astype(mx.float16)
        else:
            NP = IN // 32
            blk = self.qblk.reshape(OUT, NP, 24)
            cb = blk[:, :, :16].reshape(OUT, NP * 2, 8)     # блок-локальные
            q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.float16)
            sh = mx.arange(8, dtype=mx.uint8)
            for off, w in ((16, 16.0), (20, 32.0)):
                bits = (blk[:, :, off:off + 4].reshape(OUT, -1)[..., None]
                        >> sh) & 1
                q = q + bits.reshape(OUT, NP * 2, 16).astype(mx.float16) * w
            q = q - 32.0
        s = (mx.view(self.qs, mx.int8).astype(mx.float32).reshape(OUT, self.NSB, 16)
             * self.d.astype(mx.float32)[..., None]).astype(mx.float16)
        return (q.reshape(OUT, self.NB, 16) * s.reshape(OUT, self.NB, 1)) \
            .reshape(OUT, IN)

    def __call__(self, x):
        lead_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).astype(mx.float32)
        N = x2d.shape[0]
        if N >= GEMM_MIN_BATCH_NB:
            w = self._dequant_w()
            out = mx.matmul(x2d.astype(mx.float16), w.T).astype(mx.float32)
            return out.reshape(*lead_shape, self.out_features)
        outs, i = [], 0
        while i < N:
            c = min(NB_MAX, N - i)
            outs.append(self._gemv(x2d[i:i + c], c))
            i += c
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        return out.reshape(*lead_shape, self.out_features)

    def can_fuse_with(self, other):
        return (self.bits == other.bits
                and self.in_features == other.in_features
                and self.out_features == other.out_features)

    def _gemv(self, x2d, c):
        NSG, RS = self.cfg_override or _cfg(self.in_features,
                                            self.out_features)
        n_tg = self.out_features // (NSG * RS)
        if self.bits == 8:
            kern = _get_kernel_sym8(self.in_features, self.out_features,
                                    NSG, RS, c)
            inputs = [x2d, self.qblk, self.qs, self.d]
        else:
            xbsum = mx.sum(x2d.reshape(c, self.NB, 16), axis=2)
            kern = _get_kernel_sym6(self.in_features, self.out_features,
                                    NSG, RS, c)
            inputs = [x2d, self.qblk, self.qs, self.d, xbsum]
        return kern(
            inputs=inputs,
            grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
            output_shapes=[(c, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]


class SymQuantLinearFused:
    """Фьюз K однотипных SymQuantLinear (r/k/v proj) в один launch.

    Аналог GwQuantLinearFused для раскладки sym, и он нужен не ради
    симметрии: `GwQuantLinearFused` требует `GwQuantLinear`, поэтому при
    proj в sym фьюз просто НЕ СТРОИЛСЯ (не падал -- молча не строился), и
    документированные −0.8 мс на шаге целевой пресет не получал вовсе.

    Механика ровно та же: строки K матриц конкатенируются (формат
    нетронут, ни один буфер не пересчитывается), а кернель выбирает вход
    по номеру строки через OUT_PER. Порядок суммирования внутри строки не
    меняется, поэтому фьюз обязан быть БИТ-В-БИТ равен K отдельным
    вызовам, и гейт tests/test_sym_fuse_parity.py требует именно этого.

    Только decode-путь: __call__(xstack [K, IN]) -> [K, out_per].
    Цена -- копия буферов поверх оригиналов (оригиналы нужны нефьюзнутому
    пути и GEMM-префиллу), поэтому строится он лениво, вместе со всем
    остальным фьюзом (см. QuantTMix._build_fused).
    """

    def __init__(self, lins):
        l0 = lins[0]
        assert all(isinstance(l, SymQuantLinear) for l in lins)
        assert all(l0.can_fuse_with(l) for l in lins)
        self.K = len(lins)
        self.bits = l0.bits
        self.out_per = l0.out_features
        self.out_features = self.out_per * self.K
        self.in_features = l0.in_features
        self.NB, self.NSB = l0.NB, l0.NSB
        self.qblk = mx.concatenate([l.qblk for l in lins], axis=0)
        self.qs = mx.concatenate([l.qs for l in lins], axis=0)
        self.d = mx.concatenate([l.d for l in lins], axis=0)
        mx.eval(self.qblk, self.qs, self.d)
        self.cfg_override = None

    def __call__(self, xstack):
        # xstack: [K, IN] fp32
        NSG, RS = self.cfg_override or _cfg(self.in_features, self.out_per)
        n_tg = self.out_features // (NSG * RS)
        if self.bits == 8:
            kern = _get_kernel_sym8(self.in_features, self.out_features,
                                    NSG, RS, 1, self.out_per)
            inputs = [xstack, self.qblk, self.qs, self.d]
        else:
            xbsum = mx.sum(xstack.reshape(self.K, self.NB, 16), axis=2)
            kern = _get_kernel_sym6(self.in_features, self.out_features,
                                    NSG, RS, 1, self.out_per)
            inputs = [xstack, self.qblk, self.qs, self.d, xbsum]
        out = kern(
            inputs=inputs,
            grid=(n_tg * NSG * 32, 1, 1), threadgroup=(NSG * 32, 1, 1),
            output_shapes=[(1, self.out_features)],
            output_dtypes=[mx.float32],
        )[0]
        return out.reshape(self.K, self.out_per)
