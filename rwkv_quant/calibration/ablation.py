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


def combined_sanity_check(model, data, best_bits, outlier_fracs, clip_percentiles,
                           baseline_ppl=None, ppl_threshold_pct=5.0, explosion_multiplier=5.0,
                           verbose=True):
    """
    Комбинаторная проверка перед тем, как calibrate() вернёт финальный
    QuantConfig: single-group ablation оценивает каждую группу НЕЗАВИСИМО
    и физически не может поймать эффекты ВЗАИМОДЕЙСТВИЯ при одновременном
    квантовании нескольких групп. Пример из практики: все четыре LoRA-ветки
    (w/a/v/g) на INT4 по отдельности безобидны (Δppl < 1%), но вместе дают
    ~150x взрыв ppl на rwkv7-g1h-1.5b (11.4 -> 1708) -- см. presets.py.

    Собирает финальный config целиком, меряет ppl НА НЁМ (не по группам),
    и если результат взорвался сильнее разумного запаса на ожидаемый
    комбинаторный эффект (explosion_multiplier * ppl_threshold_pct) --
    откатывает САМУЮ агрессивно квантованную группу на шаг вверх по
    битности и повторяет, пока не уложится в допуск или не кончится
    бюджет попыток. Мутирует best_bits/outlier_fracs на месте и
    возвращает их же для читаемости на стороне вызова.
    """
    baseline_ppl = baseline_ppl if baseline_ppl is not None else perplexity(model, data, QuantConfig())
    BITS_LADDER = [2, 3, 4, 6, 8, 16]
    explosion_threshold = ppl_threshold_pct * explosion_multiplier

    def combined_ppl():
        cfg = QuantConfig(clip_percentiles=clip_percentiles, outlier_fracs=outlier_fracs, **best_bits)
        return perplexity(model, data, cfg)

    ppl = combined_ppl()
    delta = 100 * (ppl - baseline_ppl) / baseline_ppl
    if verbose:
        print(f"\n  [combined sanity check] Δppl(целиком)={delta:+.2f}%  (допуск {explosion_threshold:.1f}%)")

    guard = 0
    while delta > explosion_threshold and guard < 20:
        guard += 1
        candidates = [g for g in best_bits if best_bits[g] < 16]
        if not candidates:
            break
        g = min(candidates, key=lambda g: best_bits[g])
        cur = best_bits[g]
        nxt = next((b for b in BITS_LADDER if b > cur), 16)
        if verbose:
            print(f"  [combined sanity check] откатываю {g}: INT{cur} -> INT{nxt} (самая агрессивная группа)")
        best_bits[g] = nxt
        if nxt >= 8:
            outlier_fracs.pop(g, None)
        ppl = combined_ppl()
        delta = 100 * (ppl - baseline_ppl) / baseline_ppl
        if verbose:
            print(f"    -> Δppl(целиком)={delta:+.2f}%")

    if delta > explosion_threshold and verbose:
        print(f"  [combined sanity check] ВНИМАНИЕ: не уложился в допуск за {guard} попыток -- "
              f"проверьте конфиг вручную")

    return best_bits, outlier_fracs, ppl, delta


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
