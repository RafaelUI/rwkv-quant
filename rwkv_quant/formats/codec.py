"""
Битовая раскладка .rwkvq и деквантование -- БЕЗ torch, только numpy.

Это нормативная реализация формата. Всё, что описано в докстринге
schema.py как раскладка на диске, живёт здесь; schema.py оставляет от
себя тонкие torch-обёртки, reader.py делегирует сюда деквант. Один
источник истины: если раскладка меняется, она меняется в одном файле.

Зачем отдельный модуль. Потребители формата -- rwkv-metal (MLX) и
SwiftRWKV -- принципиально torch-free, и до сих пор были вынуждены
портировать распаковку по докстрингам и сверяться с эталоном вручную.
Теперь порт делается с этих функций, а гейт tests/test_codec_parity.py
доказывает, что они бит-в-бит равны прежнему torch-пути.

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
  asym   блок gs, fp32 scale/min на блок, контейнер uint8/int8. LoRA.
  rtn    per-row scale, коды int8 либо biased split-нибблы при bits<=4,
         опциональная разреженная SpQR-надстройка поверх.
"""
import numpy as np

__all__ = [
    "pack_int4", "unpack_int4",
    "pack6", "unpack6",
    "pack_nib_block", "unpack_nib_block",
    "pack_bitplane", "unpack_bitplane",
    "dequant_sb6", "dequant_asym", "dequant_rtn",
]


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

def dequant_sb6(codes_packed, qsqm, d, dm, *, shape, gs, sb,
                qh=None, qh2=None) -> np.ndarray:
    """Каноническая sb6-раскладка -> float32 [OUT, IN].

    codes_packed  uint8  [OUT, IN/2]   блок-локальные нибблы, младшие 4 бита
    qh, qh2       uint8  [OUT, IN/8]   битплоскости 5-го и 6-го бита кода
    qsqm          uint8  [OUT, NSB, 12] 8 qs и 8 qm по 6 бит (qm со сдвигом +31)
    d, dm         fp16   [OUT, NSB]    супер-scale для qs и qm

    Формула ровно как в кернеле: s = half(qs * d), m = half(qm * dm),
    w = q * s + m. Клип scale снизу -- как в writer (см. NaN-примечание там)."""
    OUT, IN = shape
    NB = IN // gs
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


def dequant_asym(codes, gw_scale, gw_min, *, shape, gs) -> np.ndarray:
    """gw-asym (LoRA @6, gw64) -> float32 [OUT, IN].

    codes -- uint8/int8-контейнер с UNSIGNED кодами, scale/min -- fp32 на
    блок. gw_scale/gw_min могут иметь лишние колонки (NBpad от выравнивания
    блоков): индексируемся по IN, хвост игнорируется."""
    OUT, IN = shape
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
