"""
Готовые пресеты quick-start API. Калиброваны эмпирически на rwkv7-g1h-1.5b
(см. examples/ и историю ablation в calibration/). Битность/outlier_frac —
отправная точка, не гарантия для другого масштаба модели: на RWKV-7
чувствительность к квантованию НЕ переносится линейно между размерами
(например, group 'small' безобидна на 61M, но катастрофична на 1.5B без
outlier-обработки) -- см. README. Для продакшена рекомендуется
scripts/calibrate.py на целевом чекпоинте перед выбором пресета.

Оба пресета ниже зафиксированы по итогам сессий 19.07-5/6/7 (см.
NEXT_SESSION.md) -- это не первая калибровка "на глаз", а Парето-точки,
подтверждённые прямым замером ppl на eval_corpus_world.pt[:8], одна и та
же схема квантования (asym_sb6[_aw], блок 32/суперблок 8), REDUCTION --
буквально "COMPRESSION с большей битностью кода" (аналогия MXFP4/MXFP8:
один и тот же блочный scale-механизм, разная точность мантиссы), а не
другая техника ради лучшего числа.

ОБЕ схемы требуют activation-статистику (act_stats_path) для AW-критерия
поиска -- см. tests/collect_act_stats.py (~28с сборки, /tmp/act_stats_1p5b.pt
не переживает перезагрузку -- пересобрать перед использованием, если файла
нет).
"""
from .calibration.group_config import QuantConfig

# REDUCTION v2: цель -- деградация около нуля (для QAT/QLoRA-базы, где
# training чувствителен даже к небольшим потерям -- см. сессию 19.07-5).
# proj=6 БЕЗ AW (asym_sb6 plain): на 6 битах AW-взвешивание для proj
# ПЕРЕВОРАЧИВАЕТ знак выигрыша (сессия 19.07-5, изоляция по стадиям
# конвейера) -- контринтуитивно, но подтверждено замером, не теорией.
# emb_head=6 И cmix=6 -- С AW (там оно по-прежнему помогает на этой
# битности), proj=6 -- БЕЗ AW (asym_sb6). Это REDUCTION v2 (сессия
# 19.07): ppl 11.4438 vs bf16 11.430 (+0.12%, перевалидировано на
# РЕАЛЬНОМ пути real_gw=True), 1255.9MB (2.35x меньше bf16).
# int6 полностью деплоится: writer пакует 2-ю битплоскость qh2,
# кернель (quant_linear_gw.py, кернель-3) декодит int4/5/6 бит-в-бит
# с writer'ом; decode ~17.7 мс/ток на M4 base (A/B 19.07).
REDUCTION = QuantConfig(
    proj=6, cmix=6, emb_head=6,
    w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
    outlier_fracs={},
    group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                 "w_lora": 64, "a_lora": 64, "v_lora": 64},
    group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6_aw",
                      "emb_head": "asym_sb6_aw"},
    act_stats_path="/tmp/act_stats_1p5b.pt",
)

# COMPRESSION: "чемпион" из сессий 17.07-18.07, ppl 11.710 vs bf16 11.430
# (+2.4%), ~970.7MB (3.04x меньше bf16 2953MB). group-wise asym_sb6_aw
# (блок 32, суперблок 8, 6-бит qs/qm scale/min, AW-взвешенный поиск) --
# ПОЛНОСТЬЮ деплоится (bits 4/5, реальный кернель есть, real_gw=True
# работает, backends/metal/quant_linear_gw.py). Заменяет старый
# per-row+SpQR COMPRESSION (ppl 18.15/+59%, ~1181MB) -- при похожем
# размере вдвое меньшая деградация за счёт groupwise-квантования вместо
# per-row (см. README/NEXT_SESSION: "ГРАНУЛЯРНОСТЬ ВАЖНЕЕ БИТНОСТИ").
# Проверено сессией 19.07-7: дальнейшее ужатие (напр. до 4x от bf16)
# НЕДОСТИЖИМО в текущем нибл-формате даже ценой ухода за 5% ppl --
# cmix (47% размера) уже на полу int4 (INT3 в нибл-контейнере не
# экономит байт). Эта точка близка к границе Парето-фронта формата, не
# под-оптимизирована -- дальше только принципиально новая упаковка
# (суб-ниббл/sparsity), не тюнинг битности.
COMPRESSION = QuantConfig(
    proj=5, cmix=4, emb_head=5,
    w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
    outlier_fracs={},
    group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                 "w_lora": 64, "a_lora": 64, "v_lora": 64},
    group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                      "emb_head": "asym_sb6_aw"},
    act_stats_path="/tmp/act_stats_1p5b.pt",
)

PRESETS = {"reduction": REDUCTION, "compression": COMPRESSION}
