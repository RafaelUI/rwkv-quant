"""Один диагностический кейс = один процесс (гарантия освобождения памяти
ОС при выходе, вместо полагания на del/gc внутри долгоживущего процесса)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

CASES = {
    # НЕ равен пресету presets.REDUCTION: LoRA-ветки тут НЕ квантованы
    # (bits=16, dense) -- отсюда ppl 11.52 против 13.15 у настоящего
    # REDUCTION (там w/a/v_lora=4, g_lora=8). Кейс для A/B бэкенда, не
    # для оценки качества пресета.
    "reduction_dense_lora": QuantConfig(proj=8, cmix=8, emb_head=8, small=8),
    "proj":       QuantConfig(proj=4, outlier_fracs={"proj": 0.02}),
    "cmix":       QuantConfig(cmix=4, outlier_fracs={"cmix": 0.02}),
    "emb_head":   QuantConfig(emb_head=4, outlier_fracs={"emb_head": 0.02}),
    "small":      QuantConfig(small=6),
    "lora":       QuantConfig(w_lora=4, a_lora=4, v_lora=4, g_lora=4),
    "proj_cmix":  QuantConfig(proj=4, cmix=4, outlier_fracs={"proj": 0.02, "cmix": 0.02}),
    "proj_cmix_embhead": QuantConfig(proj=4, cmix=4, emb_head=4,
                                      outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02}),
    "lora_g8": QuantConfig(w_lora=4, a_lora=4, v_lora=4, g_lora=8),
    "lora_g4_biasfix": QuantConfig(w_lora=4, a_lora=4, v_lora=4, g_lora=4),
    # Открытый вопрос №1, bisection: какая из четырёх LoRA-веток по отдельности
    # даёт основной вклад в разрыв real vs fake_quant (~20x на комбинации всех
    # четырёх). Остальные три ветки держим на bits=16 (dense, без потерь).
    "w_lora_only": QuantConfig(w_lora=4),
    "a_lora_only": QuantConfig(a_lora=4),
    "v_lora_only": QuantConfig(v_lora=4),
    "g_lora_only": QuantConfig(g_lora=4),
    "compression_g4_biasfix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=4,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    # Атрибуция потерь COMPRESSION (+48.3%): поднимаем ПО ОДНОЙ группе до
    # INT8, остальное как в compression_fixed. SpQR-фракции не трогаем
    # (на INT8 SpQR в шуме, см. compression_g8_spqr) -- меняется одна
    # переменная: битность группы. Реальный бэкенд, срез [:8].
    "attrib_proj8": QuantConfig(
        proj=8, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    "attrib_cmix8": QuantConfig(
        proj=4, cmix=8, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    "attrib_emb8": QuantConfig(
        proj=4, cmix=4, emb_head=8,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    "attrib_small8": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    "attrib_wav8": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=8, a_lora=8, v_lora=8, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    # Уточнения атрибуции: small dense (векторы ~150KB суммарно -- размер
    # ноль, был ли смысл в INT6?) и w/a/v dense (проверка немонотонности:
    # INT8 дал ppl ХУЖЕ INT4 -- 20.34 vs 16.95; если dense тоже хуже,
    # это интерференция ошибок групп, класс явлений бага №1).
    "attrib_small16": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=16,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    "attrib_wav16": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=16, a_lora=16, v_lora=16, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    # Разрез ВНУТРИ групп (bits_overrides): одна матрица -> INT8, остальное
    # как compression_fixed (16.947). Паттерны кроют оба naming'а.
    "inner_proj_r": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"r_proj.weight": 8, "att.receptance.weight": 8},
    ),
    "inner_proj_k": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"k_proj.weight": 8, "att.key.weight": 8},
    ),
    "inner_proj_v": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"v_proj.weight": 8, "att.value.weight": 8},
    ),
    "inner_proj_o": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"o_proj.weight": 8, "att.output.weight": 8},
    ),
    "inner_cmix_key": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"cmix.key.weight": 8, "ffn.key.weight": 8},
    ),
    "inner_cmix_val": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"cmix.value.weight": 8, "ffn.value.weight": 8},
    ),
    "inner_emb": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"emb.weight": 8},
    ),
    "inner_head": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"head.weight": 8},
    ),
    # Кандидат COMPRESSION+: топ-3 чувствительных места из inner-атрибуции
    # (cmix.value -1.47, small -1.50, head -0.80) в INT8, цена ~+268MB.
    # Аддитивность МЕЖДУ группами не гарантирована (см. attrib_wav8/16) --
    # только замер. Совпадает с рецептом Q4_K_M (Q6_K на ffn_down/attn_v).
    "compression_plus": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"cmix.value.weight": 8, "ffn.value.weight": 8, "head.weight": 8},
    ),
    # Group-wise scale прототип (per-32 асимметрично, как Q4_K; SpQR на
    # gw-группах выключен -- локальные scale поглощают выбросы сами).
    "gw32_cmix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "emb_head": 0.02},
        group_scale={"cmix": 32},
    ),
    # MXFP4-вариант gw32_cmix (развилка формата v2, №4i): тот же блок 32,
    # но E8M0-scale + E2M1 вместо асимметричного fp16 scale+min.
    # Сравнивать с gw32_cmix=15.876 и baseline compression_fixed=16.947.
    "mxfp4_cmix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "emb_head": 0.02},
        group_scale={"cmix": 32},
        group_scale_mode={"cmix": "mxfp4"},
    ),
    # MXFP4 + SpQR(2%) на cmix: выбросы в sparse, блоки квантуют остаток.
    # Последний шанс MXFP4-семантики: асимметрии у E8M0+E2M1 нет, хвосты
    # может спасти только sparse. Сравнивать с mxfp4_cmix=17.216,
    # gw32_cmix=15.876, baseline=16.947.
    "mxfp4_cmix_spqr": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        group_scale={"cmix": 32},
        group_scale_mode={"cmix": "mxfp4"},
    ),
    # Q4_K-стиль хранение scale/min: суперблок 256 (8x32), 6-битные
    # scale/min против пары fp16 на суперблок => 4.5 бит/элемент против
    # 5.0 у чистого gw32. Вопрос: сколько ppl стоит квантование scale.
    # Сравнивать с gw32_cmix=15.876.
    "gw32sb6_cmix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "emb_head": 0.02},
        group_scale={"cmix": 32},
        group_scale_mode={"cmix": "asym_sb6"},
    ),
    # sb6 + грид/LS-поиск scale/min на блоке (make_qkx2-стиль). Тот же
    # формат 4.5 бит, меняются только значения. Сравнивать с
    # gw32sb6_cmix=16.075 и gw32_cmix(fp16)=15.876.
    "gw32sb6s_cmix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "emb_head": 0.02},
        group_scale={"cmix": 32},
        group_scale_mode={"cmix": "asym_sb6_search"},
    ),
    # AW-взвешенный поиск в gw32sb6 (E[x^2] в критерии грида и LS).
    # Формат тот же 4.5 бит. Сравнивать с gw32sb6s_cmix=16.038,
    # gw32_cmix(fp16)=15.876. Остальные int4-группы идут per-row-AW
    # (act_stats_path работает на них как в aw_* кейсах №4f).
    "aw_gw32sb6_cmix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "emb_head": 0.02},
        group_scale={"cmix": 32},
        group_scale_mode={"cmix": "asym_sb6_aw"},
        # только ffn-ключи: proj/emb_head остаются на обычном per-row+SpQR,
        # чтобы сравнение с gw32sb6s_cmix изолировало эффект AW на cmix
        act_stats_path="/tmp/act_stats_ffn.pt",
    ),
    # КОМПОЗИТ -- претендент на новый COMPRESSION-пресет: полная AW-статистика
    # (per-row-AW+SpQR на proj/emb_head), gw32sb6+AW-поиск на cmix, small=8.
    # Сравнивать с aw_small8=14.017 (1181MB); cmix на 4.5 бит => ~+35MB.
    "aw_v2_composite": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={"proj": 0.02, "emb_head": 0.02},
        group_scale={"cmix": 32},
        group_scale_mode={"cmix": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # ОДНОРОДНЫЙ v2: gw32sb6+AW на ВСЕХ int4-группах (proj/cmix/emb_head),
    # SpQR полностью выключен -- проверка "один кернель, одна раскладка".
    # emb без статистики (вход -- индексы) => невзвешенный поиск, ок.
    # Сравнивать с aw_v2_composite=13.757 (~1216MB); однородный ~1251MB
    # (proj/emb_head тоже 4.5 бит вместо ~4.3 c per-row+SpQR).
    "aw_v2_uniform": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Эмуляция MLX INT6 (rwkv7-1.5B-g1g-mlx-6bit): асимметричный RTN,
    # группа 64, fp16 scale/bias -- схема совпадает с нашим gw-путём
    # (mode asym) 1:1. Отличие от оригинала: у MLX lora-up (IN=96/64)
    # остались fp16, у нас тоже 6 бит -- эффект пренебрежим. ~1.19GB.
    # Сравнивать с aw_v2_uniform=13.525 (~1251MB), REDUCTION=13.15 (1530MB).
    "mlx_int6_emu": QuantConfig(
        proj=6, cmix=6, emb_head=6,
        w_lora=6, a_lora=6, v_lora=6, g_lora=6, small=16,
        outlier_fracs={},
        group_scale={"proj": 64, "cmix": 64, "emb_head": 64,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64, "g_lora": 64},
    ),
    # Якорь: всё bf16, без квантования. Абсолютный ноль деградации.
    "bf16_baseline": QuantConfig(
        proj=16, cmix=16, emb_head=16,
        w_lora=16, a_lora=16, v_lora=16, g_lora=16, small=16,
        outlier_fracs={},
    ),
    # aw_v2_uniform, но small=16: гипотеза -- small-тензоры (decay/x_x/ln)
    # были главным источником деградации всей линейки. Цена: единицы MB.
    "aw_v2_uniform_s16": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=16,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Верифицированный int8-якорь на текущем корпусе (REDUCTION 13.15 --
    # предположительно до-корпусная эра, требовал перепроверки).
    "int8_perrow": QuantConfig(
        proj=8, cmix=8, emb_head=8,
        w_lora=8, a_lora=8, v_lora=8, g_lora=8, small=8,
        outlier_fracs={},
    ),
    "gw32_all": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32},
    ),
    "gw64_all": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        group_scale={"proj": 64, "cmix": 64, "emb_head": 64},
    ),
    # Композит: gw32 везде (int4) + INT8 на топ-чувствительных местах из
    # №4d (cmix.value, head, small). INT8-тензоры тоже идут через gw32
    # (асимметричный int8 -- строго не хуже per-row).
    "gw32_plus": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        bits_overrides={"cmix.value.weight": 8, "ffn.value.weight": 8, "head.weight": 8},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32},
    ),
    # Гибрид: gw32 ТОЛЬКО на cmix (где он доказанно лучше per-row+SpQR:
    # 15.88 vs 16.95 в изоляции), остальное как compression_plus.
    # cmix.value через override -> gw32-int8.
    "gw32_hybrid": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={"proj": 0.02, "emb_head": 0.02},
        bits_overrides={"cmix.value.weight": 8, "ffn.value.weight": 8, "head.weight": 8},
        group_scale={"cmix": 32},
    ),
    # Activation-aware (imatrix-гипотеза №4e): те же битности, что
    # compression_fixed / compression_plus, но scale и SpQR-отбор взвешены
    # E[x^2] с калибровочного среза [8:16].
    "aw_compression": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    "aw_plus": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        bits_overrides={"cmix.value.weight": 8, "ffn.value.weight": 8, "head.weight": 8},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # aw_compression + small=8: small суммарно ~150KB, размер файла тот же
    # 1181MB; кандидат на замену пресета COMPRESSION.
    "aw_small8": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Проверка устойчивости AW к калибровочному корпусу: та же конфигурация,
    # что aw_small8 (14.017 на статистике harrier-среза [8:16]), но E[x^2]
    # собран на ЛИТЕРАТУРНОМ пользовательском тексте (~/Develop/test.txt,
    # /tmp/calib_user.pt) -- другой домен. Если ppl держится ~14 --
    # взвешивание ловит структуру модели, а не специфику корпуса.
    "aw_small8_usercalib": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        act_stats_path="/tmp/act_stats_user.pt",
    ),
    "compression_fixed": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    ),
    # Кандидат на замену g_lora=8-воркэраунда: g_lora=4 + SpQR (frac=0.02),
    # раз SpQR полностью гасит межслойную нестабильность в изоляции (159.64
    # -> 12.15). Проверяем СОВОКУПНЫЙ эффект вместе с proj/cmix/emb_head/
    # small на INT4/6 -- а не g_lora в вакууме.
    "compression_g4_spqr": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=4,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02, "g_lora": 0.02},
    ),
    "compression_g4_spqr01": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=4,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02, "g_lora": 0.01},
    ),
    "compression_g8_spqr": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02, "g_lora": 0.02},
    ),
    # Итоговый конфиг из calibrate() end-to-end (открытый вопрос №2,
    # tests/diagnose_calibrate_e2e.py): g_lora=6 выбран ИСКЛЮЧИТЕЛЬНО на
    # основании fake_quant/RWKV7Ref (Δ=+0.09% на INT6 fake). Проверяем, не
    # даёт ли эта fake-оценка ложную уверенность -- по аналогии с g_lora=4,
    # где fake предсказывал +8.7%, а реальный пайплайн дал +1363%.
    "calibrated_e2e": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=6,
        small=6,
        outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
        clip_percentiles={"small": 99.9},
    ),
    # ЛЕСТНИЦА БИТНОСТИ (открытый вопрос №1): разрыв 13.53 (4.5bpw,
    # aw_v2_uniform) <-> 11.44 (6.5bpw, mlx_int6_emu). Ищем точку ~5-5.5bpw
    # около +2-5%. База -- aw_v2_uniform (gw32sb6+AW, без SpQR).
    # bpw больших групп: gw32sb6 = bits+0.5; gw64sb6 = bits+0.25.
    # int5: тело на 5 битах, gw32sb6+AW -> 5.5 bpw больших групп.
    "ladder_int5": QuantConfig(
        proj=5, cmix=5, emb_head=5,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # int5 + gw64: гранулярность vs битность на 5 битах -> 5.25 bpw.
    "ladder_int5_gw64": QuantConfig(
        proj=5, cmix=5, emb_head=5,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 64, "cmix": 64, "emb_head": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Тело int4, чувствительные тензоры (№4d: cmix.value, head) на 6 бит
    # тем же gw32sb6-путём -> ~4.7-4.8 bpw.
    "ladder_int4_sens6": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        bits_overrides={"cmix.value.weight": 6, "ffn.value.weight": 6,
                        "head.weight": 6},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Изоляция LoRA-вклада: тело как aw_v2_uniform (int4), но w/a/v_lora
    # на 6 бит gw64 (как в lossless mlx_int6_emu) вместо per-row int4.
    # Если разрыв 13.53<->11.44 сидит в LoRA -- увидим сильный сдвиг вниз
    # при цене ~0 MB (LoRA крошечные).
    "ladder_lora6": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),    # Потолок LoRA-гипотезы: LoRA полностью dense fp16, тело aw_v2_uniform.
    # Разница с ladder_lora6 покажет, есть ли что добирать сверх 6 бит.
    "ladder_lora16": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=16, a_lora=16, v_lora=16, g_lora=16, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Композит: int5-тело + LoRA gw64@6 (закон №5 -- только замером).
    # int4-тело + lora6 + чувствительные (cmix.value, head) на 6 бит:
    # дешёвое закрытие части остатка 12.03->11.44 без общего int5.
    "ladder_int4_sens6_lora6": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        bits_overrides={"cmix.value.weight": 6, "ffn.value.weight": 6,
                        "head.weight": 6},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # int5 gw64 + lora6: -0.25 bpw против ladder_int5_lora6 (11.676).
    "ladder_int5_gw64_lora6": QuantConfig(
        proj=5, cmix=5, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 64, "cmix": 64, "emb_head": 64,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    "ladder_int5_lora6": QuantConfig(
        proj=5, cmix=5, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Атрибуция бит-бюджета (Δppl/MB): тело int4, LoRA gw64@6 (якорь
    # ladder_lora6=12.030), ОДНА группа поднята до 5 бит.
    # Цена бита: cmix 100.7MB, proj 50.3MB, emb_head 33.6MB.
    "l6_proj5": QuantConfig(
        proj=5, cmix=4, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Атрибуция бит-бюджета (Δppl/MB): тело int4, LoRA gw64@6 (якорь
    # ladder_lora6=12.030), ОДНА группа поднята до 5 бит.
    # Цена бита: cmix 100.7MB, proj 50.3MB, emb_head 33.6MB.
    "l6_cmix5": QuantConfig(
        proj=4, cmix=5, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Атрибуция бит-бюджета (Δppl/MB): тело int4, LoRA gw64@6 (якорь
    # ladder_lora6=12.030), ОДНА группа поднята до 5 бит.
    # Цена бита: cmix 100.7MB, proj 50.3MB, emb_head 33.6MB.
    "l6_emb5": QuantConfig(
        proj=4, cmix=4, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Проверка аддитивности: proj5+emb5, cmix4. Прогноз ~11.67 при -100MB vs int5_lora6.
    "l6_proj5_emb5": QuantConfig(
        proj=5, cmix=4, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # cmix вниз до int3 (gw32sb6+AW, 3.5bpw): экономия ~100MB, вопрос -- цена в ppl.
    "l6_cmix3": QuantConfig(
        proj=4, cmix=3, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Чемпион (11.668, ~961MB) + cmix на gw64 (4.25bpw): -25MB, вопрос цены.
    "champ_cmixgw64": QuantConfig(
        proj=5, cmix=4, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 64, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Расщепление emb_head: группа на 4, только head.weight -> 5 через
    # override (чувствителен head, не emb). -17MB против чемпиона.
    "champ_headonly5": QuantConfig(
        proj=5, cmix=4, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        bits_overrides={"head.weight": 5},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Чемпион + cmix.key -> int3 (value остаётся 4): -50MB, чувствительным
    # по №4d был value; проверяем, переживёт ли key.
    "champ_cmixkey3": QuantConfig(
        proj=5, cmix=4, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        bits_overrides={"cmix.key.weight": 3, "ffn.key.weight": 3},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # --- REDUCTION v2 (сессия 19.07-5): почти-lossless кандидат для
    # QAT/QLoRA-базы. Гипотеза из "Открытых вопросов" №2: proj6/emb_head6,
    # cmix держим на 4 (насыщен по №"Ключевые выводы 18.07-вечер":
    # cmix+1бит = -0.003 ppl при int4 gw32sb6+AW, то есть почти бесплатно
    # остаться на 4). Референсные точки уже известны: чемпион (proj5/
    # cmix4/emb5) = 11.710 (+2.4%); uniform 6-бит ВСЕ группы (plain asym
    # gw64, НЕ asym_sb6_aw) = 11.439 (+0.07%, ~1190MB) -- уже почти
    # lossless, но крупнее. Проверяем: держит ли AW-схема (asym_sb6_aw,
    # та же, что дала чемпиону эффективность) 6-бит proj/emb_head при
    # cmix=4 -- и близко ли это к uniform-6bit по качеству при меньшем
    # размере. Изолированные бампы (p6e5/p5e6) -- проверка аддитивности
    # (закон №5, "предполагать нельзя -- только замер"), по аналогии с
    # тем, как была подтверждена аддитивность proj5+emb5 для чемпиона.
    "reduction_v2_p6e5c4": QuantConfig(
        proj=6, cmix=4, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    "reduction_v2_p5e6c4": QuantConfig(
        proj=5, cmix=4, emb_head=6,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    "reduction_v2_p6e6c4": QuantConfig(
        proj=6, cmix=4, emb_head=6,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),

    # --- Изоляция proj@6 по стадиям конвейера (аналог "Изоляция cmix"
    # 18.07): почему proj 5->6 внутри asym_sb6_aw не даёт ожидаемого
    # улучшения (даже небольшой регресс, p6e5c4=11.7298 > чемпион 11.710).
    # Держим cmix/emb_head/LoRA на уровне чемпиона, варьируем ТОЛЬКО
    # схему proj: plain sb6 (без поиска) / +грид-поиск (без AW) /
    # +AW (= p6e5c4, уже есть) / плоская "asym" gs=64 (референсная схема
    # без суперблочного 6-бит scale вообще -- контроль).
    "diag6_proj_sb6plain": QuantConfig(
        proj=6, cmix=4, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    "diag6_proj_sb6search": QuantConfig(
        proj=6, cmix=4, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_search", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),


    # --- REDUCTION v2, честная сборка ВНУТРИ деплоящейся sb6-схемы (AW
    # выключен на 6-битных группах -- см. диагностику proj выше). "asym"
    # gs=64 отброшен: у него нет реального кернеля (падает в dense fp16
    # при загрузке -- см. quant_model._linear), число 11.439 нерелевантно
    # для деплоя. Изоляция emb_head@6 и cmix@5, оба БЕЗ AW (plain sb6),
    # остальное на уровне чемпиона -- по аналогии с proj.
    "diag6_emb_sb6plain": QuantConfig(
        proj=5, cmix=4, emb_head=6,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    "diag5_cmix_sb6plain": QuantConfig(
        proj=5, cmix=5, emb_head=5,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Полная сборка: proj6/emb6 БЕЗ AW (plain sb6), cmix4 остаётся с AW
    # (там AW доказанно помогает на 4 битах -- закон отдельный для
    # низких/высоких битов, не выключаем без причины).
    # ИСПРАВЛЕНО по факту замеров: AW group-specific, не bit-depth-specific
    # (emb_head: AW ПОМОГАЕТ на 6 битах -- p5e6c4 11.661 с AW vs
    # diag6_emb_sb6plain 11.775 без AW; proj: AW ВРЕДИТ на 6 битах --
    # p6e5c4 11.730 с AW vs diag6_proj_sb6plain 11.691 без AW). Поэтому
    # proj БЕЗ AW, emb_head С AW (как раньше), cmix С AW (как чемпион,
    # там 4-битный код и AW доказанно помогает).
    "reduction_v2_noaw_p6e6c4": QuantConfig(
        proj=6, cmix=4, emb_head=6,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),

    # --- Рост размера REDUCTION v2 к около-нулевой деградации (19.07-6):
    # 84MB зазор между COMPRESSION и reduction_v2_noaw_p6e6c4 за 0.5pp --
    # плохая точка на кривой. Растим cmix (самая тяжёлая группа, 805M
    # элементов) поверх proj6-noAW/emb6-AW.
    "reduction_v2_p6e6c5_mixed": QuantConfig(
        proj=6, cmix=5, emb_head=6,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    "reduction_v2_p6e6c6_mixed": QuantConfig(
        proj=6, cmix=6, emb_head=6,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),

    # --- Пол размера внутри текущего нибл-формата (19.07-7): можно ли
    # ужать COMPRESSION до 4x от bf16 (~738MB) в бюджет 5% ppl?
    # Всё на int4 (proj/emb тоже) -- минимальный размер без нового
    # суб-ниббл формата.
    "compression_floor_p4c4e4": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Тот же пол, но gs=64 для cmix (шире блок -> меньше служебных байт
    # на scale/min ценой грубее scale) -- проверка второго рычага сжатия,
    # не битности кода.
    "compression_floor_gs64cmix": QuantConfig(
        proj=4, cmix=4, emb_head=4,
        w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
        outlier_fracs={},
        group_scale={"proj": 32, "cmix": 64, "emb_head": 32,
                     "w_lora": 64, "a_lora": 64, "v_lora": 64},
        group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                          "emb_head": "asym_sb6_aw"},
        act_stats_path="/tmp/act_stats_1p5b.pt",
    ),
    # Текущий шипованный REDUCTION (presets.py) как есть -- перепроверка
    # старого замечания "ppl 13.15" (реестр 19.07-5/6).
    "reduction_current_preset": QuantConfig(
        proj=8, cmix=8, emb_head=8,
        w_lora=4, a_lora=4, v_lora=4, g_lora=8, small=8,
    ),
}


def main():
    name = sys.argv[1]
    cfg = CASES[name]

    sd = torch.load(CKPT_PTH, map_location="cpu")
    data = torch.load(CORPUS)[:8].numpy()

    tensors = {key: quantize_tensor(key, w, cfg) for key, w in sd.items()}
    ckpt = QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                                head_size=HEAD_SIZE, vocab_size=VOCAB,
                                tensors=tensors, config_repr=repr(cfg))
    del sd
    model = QuantRWKV7(ckpt)

    total_nll, total_tok = 0.0, 0
    batch_size = 4
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i + batch_size]
            idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
            logits = model(idx); mx.eval(logits)
            logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-12))
            B, T, V = logp.shape
            idxf = target.reshape(-1); logpf = logp.reshape(-1, V)
            nll = -logpf[np.arange(len(idxf)), idxf]
            total_nll += nll.sum(); total_tok += nll.size
    ppl = float(np.exp(total_nll / total_tok))
    print(f"{name:20s} ppl={ppl:14.4f}")


if __name__ == "__main__":
    main()
