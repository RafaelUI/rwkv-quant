"""
Формат .rwkvq -- портативный квантованный чекпоинт RWKV-7, независимый
от backend'а (Metal/CUDA потребляют один и тот же файл).

В отличие от calibration.fake_quant (который просто искажает bf16-тензор,
чтобы измерить ppl-эффект), здесь хранится РЕАЛЬНОЕ упакованное
представление: int8-коды + per-row scale, и отдельно -- разреженная
SpQR-надстройка (индексы + точные bf16-значения выбросов) для групп,
где она применялась.

Codes: при bits >= 5 -- int8 [out, in]. При bits <= 4 -- УПАКОВАНЫ в
uint8-нибблы (codes_packed, [out, ceil(in/2)]), BIASED SPLIT-раскладка:
байт i несёт колонку i в low-ниббле и колонку i + ceil(in/2) в high-ниббле
(при нечётном in последний high-ниббл несёт bias, т.е. код 0); в ниббле
хранится code + 8 (диапазон [0,15], БЕЗ знака). Обе особенности выбраны под
Metal-кернель: split даёт векторную распаковку (&0xF / >>4 на uchar4 против
16 скалярных знаковых сдвигов на чередовании -- то делало GEMV ALU-bound),
biased убирает знаковое расширение вовсе -- поправка sum(x*(n-8)) =
sum(x*n) - 8*sum(x) выносится из цикла одной строкой. INT2/INT3 тоже
хранятся нибблами. Поля codes / codes_packed взаимоисключающие.
Это даёт INT4-группам честную половину размера int8 и на диске, и в GPU-памяти
(packed-кернель в backends/metal/quant_linear_v2.py читает нибблы напрямую).
"""
from dataclasses import dataclass, field
import torch


@dataclass
class QuantizedTensor:
    key: str                      # исходный ключ в state_dict
    group: str                    # к какой квантуемой группе относится
    bits: int                     # 16 = не квантован, хранится as-is
    shape: tuple
    codes: torch.Tensor = None    # int8 [out,in], только если 5 <= bits < 16
    codes_packed: torch.Tensor = None  # uint8 [out,ceil(in/2)], только если bits <= 4
    scale: torch.Tensor = None    # fp16, per-row [n_rows, 1], только если bits < 16
    dense: torch.Tensor = None    # исходный тензор as-is, только если bits >= 16
    outlier_indices: torch.Tensor = None  # int32 [n_outliers, 2] (row, col), опционально
    outlier_values: torch.Tensor = None   # bf16 [n_outliers], опционально
    # --- формат v2 (groupwise), поля взаимоисключающие с per-row scale ---
    # gw_mode: "" (не gw) | "sb6" (блок gs, суперблок sb, 6-бит qs/qm против
    # d/dm fp16) | "asym" (блок gs, fp32 scale/min на блок, контейнер int8)
    gw_mode: str = ""
    gw_gs: int = 0                # ширина блока (32 для sb6, 64 для lora-asym)
    gw_sb: int = 0                # блоков в суперблоке (8)
    gw_d: torch.Tensor = None     # fp16 [OUT, NSB] -- супер-scale для qs
    gw_dm: torch.Tensor = None    # fp16 [OUT, NSB] -- супер-scale для qm
    gw_qsqm: torch.Tensor = None  # uint8 [OUT, NSB, 12] -- 8 qs + 8 qm по 6 бит
                                  # (qm хранится со сдвигом +31: unsigned 0..62)
    gw_qh: torch.Tensor = None    # uint8 [OUT, IN/8] -- битплоскость 5-го бита
                                  # (bits=5), бит c строки = старший бит кода c
    gw_scale: torch.Tensor = None # fp32 [OUT, NB] -- asym-режим (LoRA)
    gw_min: torch.Tensor = None   # fp32 [OUT, NB] -- asym-режим (LoRA)


@dataclass
class QuantizedCheckpoint:
    naming: str                    # "custom" | "world" (см. models/naming.py)
    n_layer: int
    n_embd: int
    head_size: int
    vocab_size: int
    tensors: dict = field(default_factory=dict)  # key -> QuantizedTensor
    config_repr: str = ""          # str(QuantConfig), для отладки/воспроизводимости


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """int8 [rows, cols] (значения в [-8, 7]) -> uint8 [rows, ceil(cols/2)].
    BIASED SPLIT-раскладка (см. докстринг модуля): в ниббле code + 8,
    low-ниббл байта i = колонка i, high = колонка i + ceil(cols/2)."""
    assert codes.dtype == torch.int8
    assert int(codes.min()) >= -8 and int(codes.max()) <= 7
    rows, cols = codes.shape
    if cols % 2:
        codes = torch.cat([codes, torch.zeros(rows, 1, dtype=torch.int8)], dim=1)
    half = codes.shape[1] // 2
    u = (codes.to(torch.int16) + 8).to(torch.uint8)     # biased: [0, 15]
    lo, hi = u[:, :half], u[:, half:]
    return lo | (hi << 4)


def unpack_int4(packed: torch.Tensor, n_cols: int) -> torch.Tensor:
    """Обратно к int8 [rows, n_cols] со знаковым расширением нибблов."""
    assert packed.dtype == torch.uint8
    lo = (packed & 0xF).to(torch.int16) - 8   # biased -> знаковый код
    hi = (packed >> 4).to(torch.int16) - 8
    out = torch.cat([lo, hi], dim=1).to(torch.int8)  # split-раскладка
    return out[:, :n_cols].contiguous()


def int8_codes(qt) -> torch.Tensor:
    """Универсальный доступ к кодам в int8 независимо от упаковки --
    для reader'а, референсного v1-бэкенда и тестов."""
    if qt.codes is not None:
        return qt.codes
    return unpack_int4(qt.codes_packed, qt.shape[1])


# ---------------- формат v2: упаковщики ----------------

def pack6(v: torch.Tensor) -> torch.Tensor:
    """uint8-значения 0..63, последняя размерность кратна 4 -> байты 3/4.
    Чанк из 4 значений (24 бита) -> 3 байта little-endian bitstream."""
    assert v.dtype == torch.uint8 and v.shape[-1] % 4 == 0
    x = v.to(torch.int32).reshape(*v.shape[:-1], -1, 4)
    b0 = (x[..., 0] | (x[..., 1] << 6)) & 0xFF
    b1 = ((x[..., 1] >> 2) | (x[..., 2] << 4)) & 0xFF
    b2 = ((x[..., 2] >> 4) | (x[..., 3] << 2)) & 0xFF
    return torch.stack([b0, b1, b2], dim=-1).reshape(*v.shape[:-1], -1).to(torch.uint8)


def unpack6(b: torch.Tensor, n: int) -> torch.Tensor:
    """Обратно: байты 3/4 -> uint8 0..63, n значений в последней размерности."""
    assert b.dtype == torch.uint8 and b.shape[-1] % 3 == 0
    x = b.to(torch.int32).reshape(*b.shape[:-1], -1, 3)
    v0 = x[..., 0] & 0x3F
    v1 = ((x[..., 0] >> 6) | (x[..., 1] << 2)) & 0x3F
    v2 = ((x[..., 1] >> 4) | (x[..., 2] << 4)) & 0x3F
    v3 = (x[..., 2] >> 2) & 0x3F
    out = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*b.shape[:-1], -1)
    return out[..., :n].to(torch.uint8)


def pack_nib_block(q: torch.Tensor, gs: int = 32) -> torch.Tensor:
    """БЛОК-ЛОКАЛЬНЫЙ split для gw-кодов (unsigned 0..15, БЕЗ bias):
    внутри блока из gs колонок байт j = q[j] | (q[j + gs/2] << 4),
    j = 0..gs/2-1. Один блок-32 = 16 байт = один uint4-лоад в кернеле.
    [OUT, IN] (IN % gs == 0) -> uint8 [OUT, IN/2]."""
    assert q.dtype == torch.uint8 and int(q.max()) <= 15
    OUT, IN = q.shape
    assert IN % gs == 0
    h = gs // 2
    qb = q.view(OUT, IN // gs, gs)
    return (qb[:, :, :h] | (qb[:, :, h:] << 4)).reshape(OUT, IN // 2).contiguous()


def unpack_nib_block(p: torch.Tensor, gs: int = 32) -> torch.Tensor:
    """Обратно к uint8-кодам 0..15, [OUT, IN]."""
    assert p.dtype == torch.uint8
    OUT, HB = p.shape
    h = gs // 2
    pb = p.view(OUT, HB // h, h)
    lo, hi = pb & 0xF, pb >> 4
    return torch.cat([lo, hi], dim=2).reshape(OUT, HB * 2).contiguous()


def pack_bitplane(bit: torch.Tensor) -> torch.Tensor:
    """Старшие биты int5-кодов (0/1, [OUT, IN], IN % 8 == 0) -> uint8
    [OUT, IN/8], бит (c % 8) байта (c // 8) = колонка c (little-endian)."""
    OUT, IN = bit.shape
    assert IN % 8 == 0
    b = bit.to(torch.uint8).view(OUT, IN // 8, 8)
    sh = torch.arange(8, dtype=torch.uint8)
    return (b << sh).sum(dim=2, dtype=torch.int32).to(torch.uint8)


def unpack_bitplane(p: torch.Tensor, n_cols: int) -> torch.Tensor:
    OUT = p.shape[0]
    sh = torch.arange(8, dtype=torch.uint8)
    bits = (p.unsqueeze(-1) >> sh) & 1
    return bits.reshape(OUT, -1)[:, :n_cols].contiguous()
