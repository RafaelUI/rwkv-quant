"""
Формат .rwkvq -- портативный квантованный чекпоинт RWKV-7, независимый
от backend'а (Metal/CUDA потребляют один и тот же файл).

В отличие от calibration.fake_quant (который просто искажает bf16-тензор,
чтобы измерить ppl-эффект), здесь хранится РЕАЛЬНОЕ упакованное
представление: int8-коды + per-row scale, и отдельно -- разреженная
SpQR-надстройка (индексы + точные bf16-значения выбросов) для групп,
где она применялась.

Codes хранятся как int8 (а не упаковано по 4/3/2 бита в байт) для простоты
первой версии -- это НЕ даёт полного теоретического сжатия на диске
(экономия только на dtype: int8 вместо bf16/fp32), но корректно
воспроизводит квантованные значения и не блокирует backends/ от
дальнейшей битовой упаковки при реальном инференсе.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class QuantizedTensor:
    key: str                      # исходный ключ в state_dict
    group: str                    # к какой квантуемой группе относится
    bits: int                     # 16 = не квантован, хранится as-is
    shape: tuple
    codes: torch.Tensor = None    # int8, только если bits < 16
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
