"""
Высокоуровневый API. Два входа:
  - quantize(ckpt, out, preset=...)   -- быстрый старт, готовые пресеты
  - quantize(ckpt, out, config=...)   -- полный контроль через QuantConfig
  - calibrate(ckpt, corpus)           -- автоматически подобрать QuantConfig
                                          под конкретный чекпоинт вместо
                                          пресета "с потолка" (см. README:
                                          чувствительность к квантованию НЕ
                                          переносится между масштабами модели)
"""
import torch

from .presets import PRESETS
from .calibration import GROUPS, QuantConfig
from .calibration.ablation import single_group_ablation, perplexity, combined_sanity_check
from .models.rwkv7_ref import RWKV7Ref
from .formats import save, quantize_file  # noqa: F401 (save -- публичный API)


def quantize(checkpoint_path: str, output_path: str, preset: str = "reduction",
             config: QuantConfig = None, real_gw: bool = True,
             verbose: bool = True):
    """
    Quick-start: quantize(ckpt, out, preset="compression")
    Advanced:    quantize(ckpt, out, config=QuantConfig(proj=4, ...))

    preset игнорируется, если передан config.

    real_gw=True (по умолчанию) -- реальная упаковка sb6, файл сжимается.
    real_gw=False -- fake-quant для измерения ppl: та же математика ошибки,
    но веса остаются плотными bf16 и файл НЕ уменьшается.
    """
    if config is None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, choose from {list(PRESETS)}")
        config = PRESETS[preset]

    # Потоковый путь: mmap + потензорное квантование с немедленным
    # освобождением, метаданные -- из форм тензоров. Прежняя версия
    # сначала инстанцировала RWKV7Ref ради пяти чисел (то есть поднимала
    # всю модель в bf16, 5.9 ГБ на 2.9B), потом грузила state_dict ЕЩЁ
    # РАЗ целиком -- на 16 ГБ это давало пик 9-12 ГБ и своп.
    return quantize_file(checkpoint_path, output_path, config,
                         real_gw=real_gw, verbose=verbose)


def calibrate(checkpoint_path: str, eval_corpus_path: str, device: str = "mps",
              ppl_threshold_pct: float = 5.0, spqr_outlier_frac: float = 0.02,
              small_clip_percentile: float = 99.9, verbose: bool = True) -> QuantConfig:
    """
    Прогоняет single-group ablation на конкретном чекпоинте и подбирает
    QuantConfig вместо использования пресета "с потолка".

    Эвристика (по находкам из README, не универсальный поиск):
      1. Для каждой группы -- минимальная битность из {8,6,4,2} с Δppl% <=
         ppl_threshold_pct без какой-либо доп. обработки.
      2. Если группа из {proj, cmix, emb_head} не прошла порог даже на INT8 --
         пробуем INT4 + SpQR sparse outlier (percentile-clipping для них НЕ
         пробуем: он делает эти группы хуже, см. README).
      3. Если группа "small" не прошла порог -- пробуем INT6 + percentile
         clip (SpQR для неё не нужен, clip уже полностью решает проблему).
    """
    model = RWKV7Ref(checkpoint_path, device=device, dtype=torch.bfloat16)
    data = torch.load(eval_corpus_path).to(device)

    baseline_ppl, results = single_group_ablation(
        model, data, groups=GROUPS, bits_grid=(8, 6, 4, 2), verbose=verbose)

    best_bits = {}
    for group in GROUPS:
        group_results = [r for r in results if r[0] == group and r[3] <= ppl_threshold_pct]
        best_bits[group] = min((r[1] for r in group_results), default=16)

    outlier_fracs, clip_percentiles = {}, {}

    for group in ("proj", "cmix", "emb_head"):
        if best_bits[group] > 4:
            cfg_try = QuantConfig(**{group: 4}, outlier_fracs={group: spqr_outlier_frac})
            ppl_try = perplexity(model, data, cfg_try)
            delta = 100 * (ppl_try - baseline_ppl) / baseline_ppl
            if verbose:
                print(f"  retry {group} @ INT4+SpQR({spqr_outlier_frac*100:.0f}%): Δ={delta:+.2f}%")
            if delta <= ppl_threshold_pct * 3:  # мягче: это уже "сжатие ценой качества"
                best_bits[group] = 4
                outlier_fracs[group] = spqr_outlier_frac

    if best_bits["small"] > 6:
        cfg_try = QuantConfig(small=6, clip_percentiles={"small": small_clip_percentile})
        ppl_try = perplexity(model, data, cfg_try)
        delta = 100 * (ppl_try - baseline_ppl) / baseline_ppl
        if verbose:
            print(f"  retry small @ INT6+clip(p{small_clip_percentile}): Δ={delta:+.2f}%")
        if delta <= ppl_threshold_pct:
            best_bits["small"] = 6
            clip_percentiles["small"] = small_clip_percentile

    # single-group ablation выше оценивает каждую группу НЕЗАВИСИМО и не
    # может поймать эффекты взаимодействия при одновременном квантовании
    # (см. presets.py: все 4 LoRA-ветки на INT4 порознь безобидны, вместе --
    # ~150x взрыв ppl). Обязательная проверка целиком перед выдачей конфига.
    best_bits, outlier_fracs, final_ppl, final_delta = combined_sanity_check(
        model, data, best_bits, outlier_fracs, clip_percentiles,
        baseline_ppl=baseline_ppl, ppl_threshold_pct=ppl_threshold_pct, verbose=verbose)

    config = QuantConfig(clip_percentiles=clip_percentiles, outlier_fracs=outlier_fracs, **best_bits)
    if verbose:
        print(f"\nCalibrated config: {config}\noutlier_fracs={outlier_fracs}  clip={clip_percentiles}")
        print(f"Финальный Δppl(целиком) = {final_delta:+.2f}%")
    return config
