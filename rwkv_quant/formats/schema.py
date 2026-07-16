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
