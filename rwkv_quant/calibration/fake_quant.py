"""
Функции fake-квантования весов RWKV-7: обычное symmetric per-channel RTN,
percentile-clipped вариант, и SpQR-style sparse outlier extraction.

Ключевая находка (см. README проекта): clipping спасает группы, где выброс --
это "мусорное" значение на фоне плотного нормального кластера (напр. r_k),
но ВРЕДИТ полноранговым dense-матрицам (proj/cmix), где "хвост" распределения
несёт реальный обученный сигнал. Для dense-групп нужен SpQR-style подход:
сохранить выбросы точно (разреженно, в bf16), а не резать/искажать их.
"""
import torch
import torch.nn.functional as F

from .group_config import QuantConfig


def fake_quantize_sparse_outlier(w: torch.Tensor, bits: int, outlier_frac: float) -> torch.Tensor:
    """SpQR-style: держим top outlier_frac-долю значений КАЖДОЙ строки в exact bf16
    (разреженно, поэлементно), остальное квантуем с чистой шкалой (без искажения
    выбросами). outlier_frac=0.01 -> топ-1% значений строки остаются точными."""
    orig_dtype = w.dtype
    w32 = w.float()
    n_cols = w32.shape[1]
    k = max(1, int(round(n_cols * outlier_frac)))
    abs_w = w32.abs()
    kth_val = torch.topk(abs_w, k, dim=1, largest=True).values[:, -1:].clamp_min(1e-8)
    outlier_mask = abs_w >= kth_val
    w_dense = torch.where(outlier_mask, torch.zeros_like(w32), w32)
    qmax = 2 ** (bits - 1) - 1
    amax = w_dense.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    qv = torch.clamp(torch.round(w_dense / scale), -qmax - 1, qmax)
    deq = qv * scale
    result = torch.where(outlier_mask, w32, deq)
    return result.to(orig_dtype)


def fake_quantize(w: torch.Tensor, bits: int, per_channel: bool = True, clip_percentile=None) -> torch.Tensor:
    if bits >= 16:
        return w
    orig_dtype = w.dtype
    w32 = w.float()
    qmax = 2 ** (bits - 1) - 1
    if clip_percentile is not None and per_channel and w32.dim() >= 2:
        flat = w32.abs().reshape(w32.shape[0], -1)
        clip_val = torch.quantile(flat, clip_percentile / 100, dim=1, keepdim=True).clamp_min(1e-8)
        shape = [w32.shape[0]] + [1] * (w32.dim() - 1)
        clip_val = clip_val.view(*shape)
    elif clip_percentile is not None:
        clip_val = torch.quantile(w32.abs().reshape(-1), clip_percentile / 100).clamp_min(1e-8)
    elif per_channel and w32.dim() >= 2:
        clip_val = w32.abs().amax(dim=tuple(range(1, w32.dim())), keepdim=True).clamp_min(1e-8)
    else:
        clip_val = w32.abs().amax().clamp_min(1e-8)
    scale = clip_val / qmax
    wc = torch.clamp(w32, -clip_val, clip_val)
    qv = torch.clamp(torch.round(wc / scale), -qmax - 1, qmax)
    return (qv * scale).to(orig_dtype)


def q(w, group, cfg: "QuantConfig"):
    bits = cfg.bits[group]
    if group in cfg.outlier_fracs and w.dim() >= 2:
        return fake_quantize_sparse_outlier(w, bits, cfg.outlier_fracs[group])
    return fake_quantize(w, bits, clip_percentile=cfg.clip_percentiles.get(group))
