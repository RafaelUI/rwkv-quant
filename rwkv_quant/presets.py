"""
Готовые пресеты quick-start API. Калиброваны эмпирически на rwkv7-g1h-1.5b
(см. examples/ и историю ablation в calibration/). Битность/outlier_frac —
отправная точка, не гарантия для другого масштаба модели: на RWKV-7
чувствительность к квантованию НЕ переносится линейно между размерами
(например, group 'small' безобидна на 61M, но катастрофична на 1.5B без
outlier-обработки) -- см. README. Для продакшена рекомендуется
scripts/calibrate.py на целевом чекпоинте перед выбором пресета.
"""
from .calibration.group_config import QuantConfig

# MEDIUM: near-lossless, ~2x compression. Безопасно почти всегда.
MEDIUM = QuantConfig(
    proj=8, cmix=8, emb_head=8,
    w_lora=4, a_lora=4, v_lora=4, g_lora=8,
    small=8,
)

# STRONG: ~3.5x compression ценой заметной (но не катастрофической) потери
# качества. Использует SpQR-style sparse outlier extraction на dense-группах
# (proj/cmix/emb_head) -- percentile clipping там НЕ работает (см. README).
STRONG = QuantConfig(
    proj=4, cmix=4, emb_head=4,
    w_lora=4, a_lora=4, v_lora=4, g_lora=4,
    small=6,
    outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    clip_percentiles={"small": 99.9},
)

PRESETS = {"medium": MEDIUM, "strong": STRONG}
