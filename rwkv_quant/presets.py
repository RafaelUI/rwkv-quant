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

# REDUCTION: near-lossless, ~2x compression. Безопасно почти всегда.
REDUCTION = QuantConfig(
    proj=8, cmix=8, emb_head=8,
    w_lora=4, a_lora=4, v_lora=4, g_lora=8,
    small=8,
)

# COMPRESSION: ~3.5x compression ценой заметной (но не катастрофической)
# потери качества. Использует SpQR-style sparse outlier extraction на
# dense-группах (proj/cmix/emb_head) -- percentile clipping там НЕ работает
# (см. README).
#
# g_lora=8, НЕ 4: если все четыре LoRA-ветки (w/a/v/g) квантовать в INT4
# ОДНОВРЕМЕННО, ppl взрывается в ~150x (11.4 -> 1708 на rwkv7-g1h-1.5b,
# полный прогон через реальный quant+backends/metal/quant_model, не
# fake_quant). Каждая ветка ПО ОТДЕЛЬНОСТИ безобидна на INT4 (см. README
# ablation) -- эффект чисто комбинаторный, single-group ablation его не
# ловит по построению. g_lora=8 полностью убирает взрыв (ppl 18.15,
# Δ+59% -- ожидаемая цена INT4 на dense-группах, не катастрофа).
# Если меняете эту группу битностей -- обязательно валидируйте КОМБИНАЦИЮ
# через реальный backend, не только single-group ablation.
COMPRESSION = QuantConfig(
    proj=4, cmix=4, emb_head=4,
    w_lora=4, a_lora=4, v_lora=4, g_lora=8,
    small=6,
    outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    clip_percentiles={"small": 99.9},
)

PRESETS = {"reduction": REDUCTION, "compression": COMPRESSION}
