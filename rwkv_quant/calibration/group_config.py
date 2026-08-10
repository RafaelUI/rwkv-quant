"""
QuantConfig и список квантуемых групп параметров RWKV-7.

Группы соответствуют внутреннему представлению rwkv_quant.models.rwkv7_ref
(единому для обеих схем именования чекпоинтов -- custom и official "world"):
  "proj"     -> r_proj, k_proj, v_proj, o_proj   (полноранговые R/K/V/O)
  "w_lora"   -> decay (низкоранговая LoRA-проекция)
  "a_lora"   -> in-context learning rate (низкоранговая LoRA-проекция)
  "v_lora"   -> value residual gate (низкоранговая LoRA-проекция)
  "g_lora"   -> output gate (низкоранговая LoRA-проекция)
  "small"    -> k_k, k_a, r_k (поканальные модуляционные векторы)
  "cmix"     -> channel-mix FFN (key/value)
  "emb"      -> emb.weight        \\ раньше были ОДНОЙ группой "emb_head";
  "head"     -> head.weight       / см. ниже, почему разделены

ПОЧЕМУ emb И head РАЗДЕЛЕНЫ (10.08.2026). Замер развилки QLoRA-базы
(NEXT_SESSION, tests/ablate_qlora_lora_source.py) дал побочный результат:
`qlora_all` (emb в bf16) и REDUCTION (emb квантован) дают +0.75/+0.77 на
1.5B и +1.367/+1.37 на 2.9B -- то есть КВАНТОВАНИЕ emb НЕ СТОИТ НИЧЕГО ПО
КАЧЕСТВУ на обоих масштабах. Про head такого замера нет, и оснований
считать его столь же безразличным тоже нет: head -- это логиты, а не
таблица поиска. Пока они были одной группой, битность приходилось
выбирать по худшему из двух, а это половина 29% модели.

СОВМЕСТИМОСТЬ. "emb_head" остаётся ПСЕВДОНИМОМ и раскрывается в обе
группы: QuantConfig(emb_head=6) == QuantConfig(emb=6, head=6), и то же
для group_scale / group_scale_mode / clip_percentiles / outlier_fracs.
Пресеты, скрипты в tests/ и конфиги, прочитанные из манифестов старых
.rwkvq, продолжают работать без правок -- гейт
tests/test_group_split_compat.py требует БИТ-В-БИТ совпадения выходов
до и после разделения.
"""

GROUPS = ["proj", "w_lora", "a_lora", "v_lora", "g_lora", "small", "cmix",
          "emb", "head"]

# Псевдонимы групп: имя -> кортеж настоящих групп. Раскрываются в
# QuantConfig для ВСЕХ словарных полей сразу, чтобы не осталось места, где
# псевдоним понимается наполовину.
GROUP_ALIASES = {"emb_head": ("emb", "head")}


def expand_aliases(d):
    """{группа-или-псевдоним: значение} -> {настоящая группа: значение}.

    Явная запись побеждает псевдоним: {"emb_head": 6, "emb": 4} даёт
    emb=4, head=6. Иначе порядок ключей в словаре решал бы результат
    квантования, а это ровно тот сорт зависимости, который потом не
    воспроизводится.
    """
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        for g in GROUP_ALIASES.get(k, (k,)):
            if k in GROUP_ALIASES and g in d:
                continue          # у явного ключа приоритет
            out[g] = v
    return out


class QuantConfig:
    def __init__(self, clip_percentiles=None, outlier_fracs=None,
                 bits_overrides=None, group_scale=None, group_scale_mode=None,
                 act_stats_path=None,
                 **bits_per_group):
        self.bits = {g: 16 for g in GROUPS}
        self.bits.update(expand_aliases(bits_per_group))
        self.clip_percentiles = expand_aliases(clip_percentiles)
        self.outlier_fracs = expand_aliases(outlier_fracs)
        # bits_overrides: {подстрока ключа: bits} -- точечная битность для
        # ОТДЕЛЬНЫХ матриц поверх групповой (диагностика внутри групп:
        # r/k/v/o в proj, key/value в cmix, emb vs head). Применяется
        # только в writer.quantize_tensor (реальный бэкенд); fake_quant
        # работает по группам и overrides не видит.
        self.bits_overrides = bits_overrides or {}
        # group_scale: {группа: размер блока колонок} -- ПРОТОТИП group-wise
        # scale (см. writer._groupwise_fake_dequant): тензор квантуется
        # асимметрично по блокам gs колонок и хранится ДЕКВАНТОВАННЫМ dense
        # bf16. Только для замера ppl; SpQR на таких группах не применяется.
        self.group_scale = expand_aliases(group_scale)
        # group_scale_mode: {группа: режим} -- схема блочного квантования
        # для групп из group_scale. Полный список с описаниями --
        # calibration.groupwise.GW_MODES (asym / asym_sb6 / asym_sb6_search /
        # asym_sb6_aw / sym / sym_plain / sym_aw / mxfp4); дефолт "asym".
        self.group_scale_mode = expand_aliases(group_scale_mode)
        # act_stats_path: путь к {key: E[x^2] по входным каналам} (см.
        # tests/collect_act_stats.py). Если задан, writer квантует тензоры
        # с имеющейся статистикой activation-aware (взвешенный RTN +
        # взвешенный отбор SpQR-выбросов); без статистики -- обычный путь.
        self.act_stats_path = act_stats_path

    def __repr__(self):
        # emb и head схлопываются обратно в emb_head, когда совпадают: так
        # repr пресетов не меняется от разделения групп, и диффы конфигов
        # остаются читаемыми. Расходятся -- печатаются порознь.
        items, skip = [], set()
        for alias, members in GROUP_ALIASES.items():
            vals = {self.bits[m] for m in members}
            if len(vals) == 1:
                skip.update(members)
        for g in GROUPS:
            if g in skip:
                continue
            items.append(f"{g}={self.bits[g]}")
        for alias, members in GROUP_ALIASES.items():
            if members[0] in skip:
                items.append(f"{alias}={self.bits[members[0]]}")
        r = "QuantConfig(" + ", ".join(items)
        if self.bits_overrides:
            r += ", overrides=" + repr(self.bits_overrides)
        return r + ")"

