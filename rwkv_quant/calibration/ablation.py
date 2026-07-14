"""
Обобщённый harness для perplexity-ablation: single-group сетка по битности,
смешанные конфиги, размер модели с учётом SpQR-надстройки.

Не привязан к конкретному чекпоинту -- принимает готовую модель
(rwkv_quant.models.RWKV7Ref) и токенизированный eval-корпус (LongTensor
[n_chunks, chunk_len]).
"""
import math
import time

import torch
import torch.nn.functional as F

from .group_config import GROUPS, QuantConfig
from .outlier_scan import group_param_counts


@torch.no_grad()
def perplexity(model, data: torch.Tensor, cfg: QuantConfig = None, batch_size: int = 2) -> float:
    cfg = cfg or QuantConfig()
    total_nll, total_tok = 0.0, 0
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i + batch_size]
        logits = model.forward(batch[:, :-1], cfg)
        target = batch[:, 1:]
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()
        total_tok += nll.numel()
    return torch.exp(torch.tensor(total_nll / total_tok)).item()


def model_size_bytes(counts, incols, other_count, bits_map=None, outlier_fracs=None) -> float:
    """Честная оценка размера с учётом SpQR-надстройки: базовые коды на все
    позиции (bits_map[group] бит) + разреженная надстройка для outlier-доли
    (16 бит bf16-значение + индекс в строке)."""
    bits_map = bits_map or {}
    outlier_fracs = outlier_fracs or {}
    size_bits = other_count * 16
    for g in GROUPS:
        bits = bits_map.get(g, 16)
        n = counts.get(g, 0)
        frac = outlier_fracs.get(g, 0.0)
        idx_bits = math.ceil(math.log2(max(incols[g]))) if incols.get(g) else 12
        size_bits += n * bits
        size_bits += n * frac * (16 + idx_bits)
    return size_bits / 8


def single_group_ablation(model, data, groups=None, bits_grid=(8, 4, 2), verbose=True):
    """Прогоняет perplexity для каждой группы отдельно на каждой битности
    из bits_grid, остальные группы -- unquantized (bf16)."""
    groups = groups or GROUPS
    baseline_ppl = perplexity(model, data, QuantConfig())
    if verbose:
        print(f"BASELINE  ppl={baseline_ppl:.4f}")

    results = []
    for group in groups:
        for bits in bits_grid:
            t0 = time.time()
            ppl = perplexity(model, data, QuantConfig(**{group: bits}))
            delta_pct = 100 * (ppl - baseline_ppl) / baseline_ppl
            if verbose:
                print(f"{group:10s} INT{bits}  ppl={ppl:12.4f}  Δ={delta_pct:+9.2f}%  [{time.time()-t0:.1f}s]")
            results.append((group, bits, ppl, delta_pct))
    return baseline_ppl, results


def mixed_config_report(model, data, name: str, cfg: QuantConfig, state_dict, verbose=True):
    """Оценивает один смешанный конфиг: ppl, Δppl, размер, сжатие."""
    counts, incols, other = group_param_counts(state_dict)
    baseline_size = model_size_bytes(counts, incols, other)
    baseline_ppl = perplexity(model, data, QuantConfig())

    t0 = time.time()
    ppl = perplexity(model, data, cfg)
    delta_pct = 100 * (ppl - baseline_ppl) / baseline_ppl
    size = model_size_bytes(counts, incols, other, cfg.bits, cfg.outlier_fracs)
    compression = baseline_size / size
    if verbose:
        print(f"{name}\n  ppl={ppl:12.4f}  Δ={delta_pct:+9.2f}%  "
              f"size={size/1e6:8.1f}MB  compression={compression:.2f}x  [{time.time()-t0:.1f}s]")
    return {"ppl": ppl, "delta_pct": delta_pct, "size_bytes": size, "compression": compression}
