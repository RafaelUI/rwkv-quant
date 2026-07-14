"""
Загрузка .rwkvq. Два режима:
  - load_raw(path)         -> QuantizedCheckpoint как есть (для backends/,
                               которые будут делать реальный low-bit инференс
                               напрямую на codes/scale, без деквантования)
  - load_dequantized(path) -> обычный bf16 state_dict, готовый для
                               RWKV7Ref(...) -- нужен для валидации/сравнения
                               ppl квантованной модели с оригиналом.
"""
import torch

from .schema import QuantizedCheckpoint


def load_raw(path: str) -> QuantizedCheckpoint:
    return torch.load(path, map_location="cpu", weights_only=False)


def _dequantize_one(qt) -> torch.Tensor:
    if qt.bits >= 16:
        return qt.dense
    w = (qt.codes.float() * qt.scale.float()).to(torch.bfloat16)
    if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
        rows, cols = qt.outlier_indices[:, 0].long(), qt.outlier_indices[:, 1].long()
        w[rows, cols] = qt.outlier_values
    return w


def load_dequantized(path: str) -> dict:
    ckpt = load_raw(path)
    return {key: _dequantize_one(qt) for key, qt in ckpt.tensors.items()}
