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

from .schema import (QuantizedCheckpoint, int8_codes, unpack6,
                     unpack_nib_block, unpack_bitplane)


def load_raw(path: str) -> QuantizedCheckpoint:
    return torch.load(path, map_location="cpu", weights_only=False)


def _dequantize_one(qt) -> torch.Tensor:
    if qt.bits >= 16:
        return qt.dense
    if qt.gw_mode == "sb6":
        return _dequantize_gw_sb6(qt)
    if qt.gw_mode == "asym":
        return _dequantize_gw_asym(qt)
    w = (int8_codes(qt).float() * qt.scale.float()).to(torch.bfloat16)
    if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
        rows, cols = qt.outlier_indices[:, 0].long(), qt.outlier_indices[:, 1].long()
        w[rows, cols] = qt.outlier_values
    return w


def load_dequantized(path: str) -> dict:
    ckpt = load_raw(path)
    return {key: _dequantize_one(qt) for key, qt in ckpt.tensors.items()}


def _dequantize_gw_sb6(qt) -> torch.Tensor:
    """Формат v2: восстановление в точности по формуле кернеля --
    s = half(qs * float(d_half)), m = half(qm * float(dm_half)),
    w = q * s + m; clamp scale как в writer (см. NaN-примечание там)."""
    OUT, IN = qt.shape
    gs, NB = qt.gw_gs, IN // qt.gw_gs
    q = unpack_nib_block(qt.codes_packed, gs).to(torch.float32)
    if qt.gw_qh is not None:
        q = q + unpack_bitplane(qt.gw_qh, IN).to(torch.float32) * 16.0
    if qt.gw_qh2 is not None:
        q = q + unpack_bitplane(qt.gw_qh2, IN).to(torch.float32) * 32.0
    qs = unpack6(qt.gw_qsqm[..., :6], 8).reshape(OUT, NB).to(torch.float32)
    qm = (unpack6(qt.gw_qsqm[..., 6:], 8).reshape(OUT, NB).to(torch.int16)
          - 31).to(torch.float32)
    d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)    # [OUT, NB]
    dm = qt.gw_dm.float().repeat_interleave(qt.gw_sb, dim=1)
    scale = (qs * d).half().float().clamp_min(1e-8)
    mn = (qm * dm).half().float()
    scale_c = scale.repeat_interleave(gs, dim=1)
    mn_c = mn.repeat_interleave(gs, dim=1)
    return (q * scale_c + mn_c).to(torch.bfloat16)


def _dequantize_gw_asym(qt) -> torch.Tensor:
    OUT, IN = qt.shape
    gs = qt.gw_gs
    q = qt.codes.to(torch.float32)          # uint8-контейнер, unsigned коды
    idx = torch.arange(IN) // gs
    scale_c = qt.gw_scale[:, idx]
    mn_c = qt.gw_min[:, idx]
    return (q * scale_c + mn_c).to(torch.bfloat16)
