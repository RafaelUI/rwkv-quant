"""
Формат .rwkvq целиком -- БЕЗ torch, только numpy: контейнер, битовая
раскладка, деквантование.

Это нормативная реализация. Потребители формата -- rwkv-metal (MLX) и
SwiftRWKV -- принципиально torch-free и до сих пор портировали
распаковку по докстрингам, сверяясь с эталоном вручную. Теперь порт
делается с этого файла, и он самодостаточен:

    manifest, arrays = codec.open_rwkvq("model.rwkvq")
    w = codec.dequant_key(manifest, arrays, "emb.weight")   # float32

Ни pickle, ни исполнения кода при загрузке, ни зависимости от пакета
rwkv_quant. Буферы -- виды на mmap, страницы file-backed.

Torch-двойники в schema.py и reader.py оставлены осознанно (они быстрее
в 2.6x, см. закон 16 в NEXT_SESSION.md); что обе реализации совпадают
бит-в-бит, доказывает гейт tests/test_codec_parity.py.

ГРАНИЦА ТИПОВ. Здесь нет bfloat16: numpy его не знает. Все три
`dequant_*` возвращают float32 -- это точное надмножество bf16, каст в
bf16 делает вызывающая сторона (torch в reader.py, mx.bfloat16 в
MLX-потребителе). Промежуточная арифметика умышленно ведётся в float32 и
повторяет порядок операций кернеля, включая half-роундтрип
`(qs * d).astype(float16).astype(float32)` -- без него writer и кернель
разойдутся на последнем бите мантиссы.

Раскладки (см. также поле "kind" в манифесте сайдкара):

  sb6    блок gs, суперблок sb; нибблы блок-локальным split'ом плюс
         0-2 битплоскости старших битов, 6-битные qs/qm против fp16
         супер-scale d/dm. Это КАНОНИЧЕСКАЯ дисковая раскладка, не
         K3-интерлив -- интерлив есть деталь загрузчика Metal-бэкенда.
  sym    Q6_K: блок gs=16 БЕЗ min, суперблок sb=16; масштаб блока --
         один int8-код против одной fp16 d на суперблок. Коды ЗНАКОВЫЕ.
         При bits=8 они лежат байтами в `codes`; при bits=6 -- со сдвигом
         +32 в 0..63 и той же упаковкой, что у sb6 (ниббл + две
         битплоскости), чтобы кернель переиспользовался как есть.
  asym   блок gs, fp32 scale/min на блок, контейнер uint8/int8. LoRA.
  rtn    per-row scale, коды int8 либо biased split-нибблы при bits<=4,
         опциональная разреженная SpQR-надстройка поверх.
"""
import json
import mmap as _mmap
import re

import numpy as np

# world хранит эти LoRA-матрицы транспонированными -- см. is_transposed
_RAW_LORA_WORLD = re.compile(r"^blocks\.\d+\.att\.[wavg][12]$")

__all__ = [
    "pack_int4", "unpack_int4",
    "pack6", "unpack6",
    "pack_nib_block", "unpack_nib_block",
    "pack_bitplane", "unpack_bitplane",
    "dequant_sb6", "dequant_sym", "dequant_asym", "dequant_rtn",
    "MAGIC_ZIP", "FORMAT", "FORMAT_VERSION",
    "bf16_to_f32", "read_safetensors", "open_rwkvq", "dequant_key",
    "is_transposed", "is_raw_lora_world",
    "pack_mlx_affine", "unpack_mlx_affine", "sb6_to_mlx_affine",
    "sb6_to_k3",
]

FORMAT = "rwkvq"
# 1 -- контейнер safetensors (pickle-эра версии не имела вовсе)
# 2 -- самодескрипция: per-tensor n_blocks и transposed, структурированный
#      config вместо repr(), ссылка на токенайзер. Читатель v2 понимает v1:
#      недостающие поля выводятся как раньше (n_blocks из форм буферов,
#      transposed -- по таблице имён), см. dequant_key.
FORMAT_VERSION = 2

# первые байты zip'а, который пишет torch.save -- по ним отличается
# прежний pickle-контейнер от safetensors (там первые 8 байт -- длина
# JSON-хедера, то есть небольшое число, и байт 0x50 в нём не встречается)
MAGIC_ZIP = b"PK\x03\x04"


# ---------------- нибблы per-row RTN (bits <= 4) ----------------

def pack_int4(codes: np.ndarray) -> np.ndarray:
    """int8 [rows, cols] со значениями в [-8, 7] -> uint8 [rows, ceil(cols/2)].

    BIASED SPLIT: в ниббле лежит code + 8 (диапазон [0,15], без знака),
    low-ниббл байта i несёт колонку i, high -- колонку i + ceil(cols/2).
    При нечётном cols последний high-ниббл добивается кодом 0."""
    assert codes.dtype == np.int8
    assert codes.min(initial=0) >= -8 and codes.max(initial=0) <= 7
    rows, cols = codes.shape
    if cols % 2:
        codes = np.concatenate(
            [codes, np.zeros((rows, 1), dtype=np.int8)], axis=1)
    half = codes.shape[1] // 2
    u = (codes.astype(np.int16) + 8).astype(np.uint8)
    lo, hi = u[:, :half], u[:, half:]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray, n_cols: int) -> np.ndarray:
    """Обратно к int8 [rows, n_cols] со снятием bias'а."""
    assert packed.dtype == np.uint8
    lo = (packed & 0xF).astype(np.int16) - 8
    hi = (packed >> 4).astype(np.int16) - 8
    out = np.concatenate([lo, hi], axis=1).astype(np.int8)
    return np.ascontiguousarray(out[:, :n_cols])


# ---------------- 6-битный битстрим qs/qm ----------------

def pack6(v: np.ndarray) -> np.ndarray:
    """uint8-значения 0..63, последняя размерность кратна 4 -> байты 3/4.
    Чанк из 4 значений (24 бита) -> 3 байта little-endian bitstream."""
    assert v.dtype == np.uint8 and v.shape[-1] % 4 == 0
    x = v.astype(np.int32).reshape(v.shape[:-1] + (-1, 4))
    b0 = (x[..., 0] | (x[..., 1] << 6)) & 0xFF
    b1 = ((x[..., 1] >> 2) | (x[..., 2] << 4)) & 0xFF
    b2 = ((x[..., 2] >> 4) | (x[..., 3] << 2)) & 0xFF
    out = np.stack([b0, b1, b2], axis=-1).reshape(v.shape[:-1] + (-1,))
    return out.astype(np.uint8)


def unpack6(b: np.ndarray, n: int) -> np.ndarray:
    """Обратно: байты 3/4 -> uint8 0..63, n значений в последней размерности."""
    assert b.dtype == np.uint8 and b.shape[-1] % 3 == 0
    x = b.astype(np.int32).reshape(b.shape[:-1] + (-1, 3))
    v0 = x[..., 0] & 0x3F
    v1 = ((x[..., 0] >> 6) | (x[..., 1] << 2)) & 0x3F
    v2 = ((x[..., 1] >> 4) | (x[..., 2] << 4)) & 0x3F
    v3 = (x[..., 2] >> 2) & 0x3F
    out = np.stack([v0, v1, v2, v3], axis=-1).reshape(b.shape[:-1] + (-1,))
    return out[..., :n].astype(np.uint8)


# ---------------- блок-локальные нибблы gw-кодов ----------------

def pack_nib_block(q: np.ndarray, gs: int = 32) -> np.ndarray:
    """БЛОК-ЛОКАЛЬНЫЙ split (unsigned 0..15, БЕЗ bias): внутри блока из gs
    колонок байт j = q[j] | (q[j + gs/2] << 4). Один блок-32 = 16 байт =
    один uint4-лоад в кернеле. [OUT, IN] (IN % gs == 0) -> uint8 [OUT, IN/2]."""
    assert q.dtype == np.uint8 and q.max(initial=0) <= 15
    OUT, IN = q.shape
    assert IN % gs == 0
    h = gs // 2
    qb = q.reshape(OUT, IN // gs, gs)
    out = (qb[:, :, :h] | (qb[:, :, h:] << 4)).reshape(OUT, IN // 2)
    return np.ascontiguousarray(out.astype(np.uint8))


def unpack_nib_block(p: np.ndarray, gs: int = 32) -> np.ndarray:
    """Обратно к uint8-кодам 0..15, [OUT, IN]."""
    assert p.dtype == np.uint8
    OUT, HB = p.shape
    h = gs // 2
    pb = p.reshape(OUT, HB // h, h)
    lo, hi = pb & 0xF, pb >> 4
    out = np.concatenate([lo, hi], axis=2).reshape(OUT, HB * 2)
    return np.ascontiguousarray(out)


# ---------------- битплоскости старших битов ----------------

def pack_bitplane(bit: np.ndarray) -> np.ndarray:
    """Биты 0/1 [OUT, IN] (IN % 8 == 0) -> uint8 [OUT, IN/8]:
    бит (c % 8) байта (c // 8) = колонка c (little-endian)."""
    OUT, IN = bit.shape
    assert IN % 8 == 0
    b = bit.astype(np.uint8).reshape(OUT, IN // 8, 8)
    sh = np.arange(8, dtype=np.uint8)
    return (b << sh).sum(axis=2, dtype=np.uint32).astype(np.uint8)


def unpack_bitplane(p: np.ndarray, n_cols: int) -> np.ndarray:
    """uint8 [OUT, IN/8] -> {0,1} uint8 [OUT, n_cols]."""
    assert p.dtype == np.uint8
    OUT = p.shape[0]
    sh = np.arange(8, dtype=np.uint8)
    bits = (p[..., None] >> sh) & 1
    return np.ascontiguousarray(bits.reshape(OUT, -1)[:, :n_cols])


# ---------------- деквантование ----------------

def dequant_sb6(codes_packed, qsqm, d, dm, *, shape, gs, sb, nb=None,
                qh=None, qh2=None) -> np.ndarray:
    """Каноническая sb6-раскладка -> float32 [OUT, IN].

    codes_packed  uint8  [OUT, IN/2]   блок-локальные нибблы, младшие 4 бита
    qh, qh2       uint8  [OUT, IN/8]   битплоскости 5-го и 6-го бита кода
    qsqm          uint8  [OUT, NSB, 12] 8 qs и 8 qm по 6 бит (qm со сдвигом +31)
    d, dm         fp16   [OUT, NSB]    супер-scale для qs и qm

    nb -- число блоков; None = вывести как IN // gs (верно, пока IN кратен
    gs*sb, что справедливо для всех реальных чекпоинтов). В манифесте v2
    оно записано явно, и расхождение здесь ловится ассертом, а не тихо
    портит веса.

    Формула ровно как в кернеле: s = half(qs * d), m = half(qm * dm),
    w = q * s + m. Клип scale снизу -- как в writer (см. NaN-примечание там)."""
    OUT, IN = shape
    NB = IN // gs if nb is None else nb
    assert qsqm.shape[-2] * sb == NB, (
        f"qsqm покрывает {qsqm.shape[-2] * sb} блоков, ожидалось {NB}")
    q = unpack_nib_block(codes_packed, gs).astype(np.float32)
    if qh is not None:
        q = q + unpack_bitplane(qh, IN).astype(np.float32) * 16.0
    if qh2 is not None:
        q = q + unpack_bitplane(qh2, IN).astype(np.float32) * 32.0
    qs = unpack6(qsqm[..., :6], 8).reshape(OUT, NB).astype(np.float32)
    qm = (unpack6(qsqm[..., 6:], 8).reshape(OUT, NB).astype(np.int16)
          - 31).astype(np.float32)
    d = np.repeat(np.asarray(d, dtype=np.float32), sb, axis=1)      # [OUT, NB]
    dm = np.repeat(np.asarray(dm, dtype=np.float32), sb, axis=1)
    scale = np.maximum((qs * d).astype(np.float16).astype(np.float32),
                       np.float32(1e-8))
    mn = (qm * dm).astype(np.float16).astype(np.float32)
    return q * np.repeat(scale, gs, axis=1) + np.repeat(mn, gs, axis=1)


def dequant_sym(qs, d, *, shape, gs, sb, nb=None, codes=None,
                codes_packed=None, qh=None, qh2=None) -> np.ndarray:
    """Каноническая sym-раскладка (Q6_K) -> float32 [OUT, IN].

    codes         int8   [OUT, IN]     знаковые коды, bits=8
    codes_packed  uint8  [OUT, IN/2]   младший ниббл кода СО СДВИГОМ +32
    qh, qh2       uint8  [OUT, IN/8]   битплоскости бита 4 и бита 5
    qs            int8   [OUT, NB]     код масштаба блока
    d             fp16   [OUT, NSB]    супер-scale блока масштабов

    Формула ровно как в кернеле: s = half(qs * d), w = q * s. Min нет --
    в этом вся разница с sb6, и потому нет ни второй пары квальных
    скаляров, ни клипа scale снизу: у вырожденного блока s = 0 и коды
    тоже нули, то есть w = 0 без деления на что бы то ни было.

    Различение битности -- ПО ПРИСУТСТВИЮ БУФЕРОВ, а не по полю bits:
    `codes` есть только при восьми битах, `codes_packed` -- только при
    шести. Так читателю не нужно доверять числу в манифесте, чтобы
    правильно разобрать байты."""
    OUT, IN = shape
    NB = IN // gs if nb is None else nb
    assert qs.shape[-1] == NB, (
        f"gw_qs покрывает {qs.shape[-1]} блоков, ожидалось {NB}")
    assert np.asarray(d).shape[-1] * sb == NB, (
        f"gw_d покрывает {np.asarray(d).shape[-1] * sb} блоков, "
        f"ожидалось {NB}")
    if codes is not None:
        q = codes.astype(np.float32)
    else:
        q = unpack_nib_block(codes_packed, gs).astype(np.int16)
        if qh is not None:
            q = q + unpack_bitplane(qh, IN).astype(np.int16) * 16
        if qh2 is not None:
            q = q + unpack_bitplane(qh2, IN).astype(np.int16) * 32
        q = (q - 32).astype(np.float32)          # снятие сдвига упаковки
    dd = np.repeat(np.asarray(d, dtype=np.float32), sb, axis=1)   # [OUT, NB]
    scale = (np.asarray(qs, dtype=np.float32) * dd).astype(np.float16) \
        .astype(np.float32)
    return q * np.repeat(scale, gs, axis=1)


def dequant_asym(codes, gw_scale, gw_min, *, shape, gs, nb=None) -> np.ndarray:
    """gw-asym (LoRA @6, gw64) -> float32 [OUT, IN].

    codes -- uint8/int8-контейнер с UNSIGNED кодами, scale/min -- fp32 на
    блок. ВНИМАНИЕ: блоков здесь CEIL(IN/gs), а не IN//gs -- хвостовой
    неполный блок имеет свой scale. На реальных LoRA это не редкость:
    у `blocks.N.att.w1` формы [2048, 96] при gs=64 блоков два, тогда как
    IN//gs = 1. Читатель, посчитавший число блоков делением нацело,
    молча возьмёт чужой масштаб на последних 32 колонках."""
    OUT, IN = shape
    NB = -(-IN // gs) if nb is None else nb
    assert gw_scale.shape[-1] == NB, (
        f"gw_scale покрывает {gw_scale.shape[-1]} блоков, ожидалось {NB}")
    q = codes.astype(np.float32)
    idx = np.arange(IN) // gs
    return q * np.asarray(gw_scale, dtype=np.float32)[:, idx] \
        + np.asarray(gw_min, dtype=np.float32)[:, idx]


def dequant_rtn(scale, *, shape, codes=None, codes_packed=None,
                outlier_indices=None, outlier_values=None) -> np.ndarray:
    """per-row RTN (+ опциональная SpQR-надстройка) -> float32.

    Форма произвольного ранга: writer квантует всё с dim >= 2, включая
    (1,1,C)-параметры, а per-row scale вида [d0,1,...] вещается сам.
    Упакованный вариант (codes_packed) бывает только 2-D."""
    if codes is None:
        codes = unpack_int4(codes_packed, shape[1])
    w = codes.astype(np.float32) * np.asarray(scale, dtype=np.float32)
    if outlier_indices is not None and len(outlier_indices):
        oi = np.asarray(outlier_indices)
        w[oi[:, 0], oi[:, 1]] = np.asarray(outlier_values, dtype=np.float32)
    return w


# ---------------- перекладка в родной контейнер MLX ----------------
#
# Зачем. Потребители на MLX (SwiftRWKV, rwkv-metal) сейчас разворачивают
# матрицу целиком на каждую проекцию, и это стоит 2.2x на декоде 2.9B
# против плотного bf16 (SwiftRWKV/decode-bench). Родное ядро
# `quantized_matmul` быстрее нынешнего пути в 2.1-4.9x и быстрее
# плотного в 1.3-2.1x, но ему нужен свой битовый контейнер.
#
# ПЕРЕКЛАДКА БЕЗ ПОТЕРЬ, и это не оговорка, а суть. Наша sb6-формула --
# w = q*s + m при беззнаковых кодах q и блоке 32 -- ЭТО И ЕСТЬ affine
# MLX: там ровно w = q*scale + bias с той же группой. Значит меняется
# только укладка бит, а числа остаются те же до последнего.
# ЧЕГО ДЕЛАТЬ НЕЛЬЗЯ: mx.quantize(деквантованный_вес). Он пересчитает
# scale/bias по min/max блока и потеряет калибровку -- ту самую, ради
# которой пресеты и измерялись.
#
# Раскладка проверена для 4/5/6/8 бит: tests/probe_mlx_native_packing.py.

def pack_mlx_affine(codes: np.ndarray, bits: int) -> np.ndarray:
    """Коды [..., 32] (0..2^bits-1) -> uint32 [..., bits].

    Группа из 32 кодов -- LSB-first битовый поток: поле позиции p
    начинается на глобальном бите p*bits и переходит границу 32-битного
    слова без выравнивания. На группу уходит ровно bits слов."""
    assert codes.shape[-1] == 32, "группа MLX здесь всегда 32"
    words = np.zeros((*codes.shape[:-1], bits), dtype=np.uint32)
    c = codes.astype(np.uint32)
    for p in range(32):
        start = p * bits
        w0, off = start // 32, start % 32
        lo = min(bits, 32 - off)
        hi = bits - lo
        words[..., w0] |= ((c[..., p] & ((1 << lo) - 1)) << off).astype(np.uint32)
        if hi:
            words[..., w0 + 1] |= ((c[..., p] >> lo) & ((1 << hi) - 1)).astype(np.uint32)
    return words


def unpack_mlx_affine(words: np.ndarray, bits: int, nb: int) -> np.ndarray:
    """Обратно: uint32 [..., NB*bits] -> коды [..., NB, 32]."""
    w = words.reshape(*words.shape[:-1], nb, bits).astype(np.uint32)
    out = np.zeros((*w.shape[:-1], 32), dtype=np.uint32)
    for p in range(32):
        start = p * bits
        w0, off = start // 32, start % 32
        lo = min(bits, 32 - off)
        hi = bits - lo
        v = (w[..., w0] >> off) & ((1 << lo) - 1)
        if hi:
            v = v | ((w[..., w0 + 1] & ((1 << hi) - 1)) << lo)
        out[..., p] = v
    return out


def sb6_to_mlx_affine(codes_packed, qsqm, d, dm, *, shape, gs, sb, nb=None,
                      qh=None, qh2=None):
    """sb6 -> (wq uint32, scales fp16, biases fp16, bits) для
    mx.quantized_matmul(group_size=32).

    Числа не пересчитываются: коды берутся как есть, scale и min
    считаются ровно той же формулой, что в dequant_sb6, и уже являются
    fp16 по построению (half-роундтрип внутри).

    ЕДИНСТВЕННОЕ РАСХОЖДЕНИЕ -- вырожденные блоки. Writer клипует scale
    снизу числом 1e-8, а в fp16 оно не представимо (минимальная
    субнормаль ~6e-8) и становится нулём. У блока, где qs*d ушло под
    fp16, в контейнере окажется scale = 0 против 1e-8 в эталоне; все
    веса такого блока равны bias, и разница по модулю не превышает
    63 * 1e-8 = 6.3e-7. На REDUCTION/1.5B таких блоков ноль, но
    закладываться на это нельзя -- гейт их считает и печатает
    (tests/test_mlx_affine_repack.py)."""
    assert gs == 32, f"родная группа MLX здесь 32, а не {gs}"
    OUT, IN = shape
    NB = IN // gs if nb is None else nb
    bits = 4 + (0 if qh is None else 1) + (0 if qh2 is None else 1)

    q = unpack_nib_block(codes_packed, gs).astype(np.uint32)
    if qh is not None:
        q = q + unpack_bitplane(qh, IN).astype(np.uint32) * 16
    if qh2 is not None:
        q = q + unpack_bitplane(qh2, IN).astype(np.uint32) * 32

    qs = unpack6(qsqm[..., :6], 8).reshape(OUT, NB).astype(np.float32)
    qm = (unpack6(qsqm[..., 6:], 8).reshape(OUT, NB).astype(np.int16)
          - 31).astype(np.float32)
    dd = np.repeat(np.asarray(d, dtype=np.float32), sb, axis=1)
    ddm = np.repeat(np.asarray(dm, dtype=np.float32), sb, axis=1)
    scales = np.maximum((qs * dd).astype(np.float16).astype(np.float32),
                        np.float32(1e-8)).astype(np.float16)
    biases = (qm * ddm).astype(np.float16)

    wq = pack_mlx_affine(q.reshape(OUT, NB, gs), bits).reshape(OUT, NB * bits)
    return wq, scales, biases, bits


# ---------------- K3-интерлив (раскладка загрузчика) ----------------
#
# Metal-ядро sb6 читает не дисковую раскладку, а K3-интерлив: коды и
# масштабы ОДНОГО блока лежат рядом, отчего блок берётся 3-4
# транзакциями вместо 7. До сих пор интерлив умел строить только
# export_mlx (torch), и потребители возили рядом с .rwkvq отдельный
# сайдкар. Здесь то же самое на numpy -- после этого сайдкар не нужен
# никому: rwkv-metal и SwiftRWKV строят K3 при загрузке сами.
#
# Раскладка (источник -- backends/metal/quant_linear_gw.py):
#   qblk[row, blk] = 16Б нибблов [+4Б qh] [+4Б qh2]
#   qsqm[row, blk] = (qs, qm) двумя байтами, qm как int8 БЕЗ сдвига +31
#   ddm[row, sblk] = (d, dm) двумя fp16
#
# Цена интерлива -- +0.125 бит/вес: qsqm занимает 2 байта на блок
# вместо канонических 1.5. На 2.9B это +45 МБ В ПАМЯТИ (не на диске).

def sb6_to_k3(codes_packed, qsqm, d, dm, *, shape, gs, sb, nb=None,
              qh=None, qh2=None):
    """sb6 (дисковая раскладка) -> (qblk, qsqm_k3, ddm, xbits).

    Числа те же: это перестановка байт, а не пересчёт."""
    assert gs == 32, f"K3 определён для блока 32, не {gs}"
    OUT, IN = shape
    NB = IN // gs if nb is None else nb
    xbits = (0 if qh is None else 1) + (0 if qh2 is None else 1)

    parts = [codes_packed.reshape(OUT, NB, 16)]
    if qh is not None:
        parts.append(qh.reshape(OUT, NB, 4))
    if qh2 is not None:
        parts.append(qh2.reshape(OUT, NB, 4))
    qblk = np.ascontiguousarray(
        np.concatenate(parts, axis=2).reshape(OUT, -1))

    qs = unpack6(qsqm[..., :6], 8).reshape(OUT, NB)
    # -31 применяется ЗДЕСЬ и хранится как int8: ядро читает байт через
    # as_type<char> и повторного сдвига не делает
    qm = (unpack6(qsqm[..., 6:], 8).reshape(OUT, NB).astype(np.int16)
          - 31).astype(np.int8)
    qsqm_k3 = np.ascontiguousarray(
        np.stack([qs, qm.view(np.uint8)], axis=-1).reshape(OUT, -1))

    ddm = np.ascontiguousarray(
        np.stack([np.asarray(d), np.asarray(dm)], axis=-1).reshape(OUT, -1))
    return qblk, qsqm_k3, ddm, xbits


# ---------------- контейнер: чтение .rwkvq без torch ----------------
#
# Всё, что ниже, -- полная читалка формата на numpy. Порт в Swift или
# C++ делается с неё: разобрать safetensors (u64-длина хедера, JSON,
# сырые буферы), достать манифест из "__metadata__", позвать нужный
# dequant_*. Ни pickle, ни исполнения кода, ни зависимости от torch.

# safetensors-имена типов -> numpy. BF16 numpy не знает, поэтому едет как
# uint16 и разворачивается через bf16_to_f32 (сдвиг на 16 бит -- точный,
# bf16 есть усечённый fp32).
_ST_DTYPE = {
    "BOOL": np.bool_, "U8": np.uint8, "I8": np.int8, "I16": np.int16,
    "U16": np.uint16, "F16": np.float16, "BF16": np.uint16,
    "I32": np.int32, "U32": np.uint32, "F32": np.float32,
    "F64": np.float64, "I64": np.int64, "U64": np.uint64,
}


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """bf16, приехавший как uint16, -> float32. Точно и без таблиц:
    bf16 -- это старшие 16 бит fp32, младшие нули."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def read_safetensors(path: str):
    """(метаданные, {имя: ndarray}) через mmap, без копирования.

    Массивы -- ВИДЫ на отображённый файл: страницы file-backed, ядро
    вытесняет их само. Именно ради этого контейнер и менялся с pickle,
    который читает всё в анонимную память до первого обращения.
    Возвращённые массивы read-only; при записи в них делать .copy()."""
    with open(path, "rb") as f:
        head8 = f.read(8)
        # Без этой проверки pickle-контейнер прежней эры читается как
        # safetensors: первые 8 байт zip'а дают длину хедера порядка
        # 10^18, и вместо внятной ошибки прилетает MemoryError.
        if head8[:4] == MAGIC_ZIP:
            raise ValueError(
                f"{path}: это pickle-контейнер (torch.save), а не "
                f"safetensors. Пересохранить: "
                f"writer.save_rwkvq(reader.load_raw(path), новый_путь)")
        n = int.from_bytes(head8, "little")
        header = json.loads(f.read(n))
        mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
    buf = np.frombuffer(mm, dtype=np.uint8)
    base = 8 + n
    meta = header.pop("__metadata__", {})
    arrays = {}
    for name, info in header.items():
        b, e = info["data_offsets"]
        dt = _ST_DTYPE[info["dtype"]]
        shape = tuple(info["shape"])
        a = buf[base + b:base + e]
        arrays[name] = (a.view(dt).reshape(shape) if a.size
                        else np.empty(shape, dt))
    return meta, arrays


def open_rwkvq(path: str):
    """(манифест, буферы) для .rwkvq в контейнере safetensors.

    Манифест -- JSON из "__metadata__"["rwkvq"]: пять чисел про модель
    плюс per-tensor запись {kind, shape, bits, group, gw_gs, gw_sb,
    fields}. Буферы адресуются как "<ключ>::<поле>"."""
    meta, arrays = read_safetensors(path)
    if "rwkvq" not in meta:
        raise ValueError(f"{path}: не .rwkvq -- в __metadata__ нет ключа rwkvq")
    manifest = json.loads(meta["rwkvq"])
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path}: формат {manifest.get('format')!r}")
    if manifest.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError(f"{path}: format_version "
                         f"{manifest['format_version']} новее {FORMAT_VERSION}")
    return manifest, arrays


def is_transposed(manifest, key) -> bool:
    """Надо ли транспонировать тензор, чтобы получить вес nn.Linear [out,in].

    Официальные (world) чекпоинты хранят LoRA-матрицы w1/w2/a1/a2/v1/v2/
    g1/g2 в СЫРОЙ ориентации [in,out] -- rwkv7_ref транспонирует их после
    загрузки, а writer квантует ДО, по сырым ключам. То есть блоки и
    per-row scale у них посчитаны вдоль хранимых строк, и потребитель
    обязан деквантовать СНАЧАЛА, транспонировать ПОТОМ. В манифесте v2
    это записано явно; для v1 выводится по той же таблице имён."""
    m = manifest["tensors"][key]
    if "transposed" in m:
        return bool(m["transposed"])
    return is_raw_lora_world(key) and manifest.get("naming") == "world"


def is_raw_lora_world(key: str) -> bool:
    """Ключ world-чекпоинта, который хранится в сырой ориентации [in,out]:
    blocks.N.att.{w,a,v,g}{1,2}. Единственное место, где эта таблица имён
    записана в коде; writer по ней проставляет поле transposed, гейт
    tests/test_manifest_selfdesc.py сверяет её с тем, что РЕАЛЬНО
    транспонирует models/rwkv7_ref.py."""
    return _RAW_LORA_WORLD.match(key) is not None


def dequant_key(manifest, arrays, key) -> np.ndarray:
    """Один тензор из открытого .rwkvq -> float32 в ТОМ ЖЕ виде, в каком
    он лежал в исходном state_dict. Каст в bf16 и транспозицию (см.
    is_transposed) делает вызывающая сторона."""
    m = manifest["tensors"][key]
    kind, shape = m["kind"], tuple(m["shape"])

    def buf(field):
        return arrays.get(f"{key}::{field}")

    if kind == "dense":
        return bf16_to_f32(buf("dense"))
    if kind == "sb6":
        return dequant_sb6(buf("codes_packed"), buf("gw_qsqm"),
                           buf("gw_d"), buf("gw_dm"),
                           shape=shape, gs=m["gw_gs"], sb=m["gw_sb"],
                           nb=m.get("n_blocks"),
                           qh=buf("gw_qh"), qh2=buf("gw_qh2"))
    if kind == "sym":
        return dequant_sym(buf("gw_qs"), buf("gw_d"),
                           shape=shape, gs=m["gw_gs"], sb=m["gw_sb"],
                           nb=m.get("n_blocks"), codes=buf("codes"),
                           codes_packed=buf("codes_packed"),
                           qh=buf("gw_qh"), qh2=buf("gw_qh2"))
    if kind == "asym":
        return dequant_asym(buf("codes"), buf("gw_scale"), buf("gw_min"),
                            shape=shape, gs=m["gw_gs"], nb=m.get("n_blocks"))
    if kind == "rtn":
        ov = buf("outlier_values")
        return dequant_rtn(buf("scale"), shape=shape,
                           codes=buf("codes"), codes_packed=buf("codes_packed"),
                           outlier_indices=buf("outlier_indices"),
                           outlier_values=None if ov is None else bf16_to_f32(ov))
    raise ValueError(f"{key}: неизвестная раскладка {kind!r}")
