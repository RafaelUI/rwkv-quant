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
  "emb_head" -> emb.weight, head.weight
"""

GROUPS = ["proj", "w_lora", "a_lora", "v_lora", "g_lora", "small", "cmix", "emb_head"]


class QuantConfig:
    def __init__(self, clip_percentiles=None, outlier_fracs=None,
                 bits_overrides=None, group_scale=None, act_stats_path=None,
                 **bits_per_group):
        self.bits = {g: 16 for g in GROUPS}
        self.bits.update(bits_per_group)
        self.clip_percentiles = clip_percentiles or {}
        self.outlier_fracs = outlier_fracs or {}
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
        self.group_scale = group_scale or {}
        # act_stats_path: путь к {key: E[x^2] по входным каналам} (см.
        # tests/collect_act_stats.py). Если задан, writer квантует тензоры
        # с имеющейся статистикой activation-aware (взвешенный RTN +
        # взвешенный отбор SpQR-выбросов); без статистики -- обычный путь.
        self.act_stats_path = act_stats_path

    def __repr__(self):
        r = "QuantConfig(" + ", ".join(f"{g}={self.bits[g]}" for g in GROUPS)
        if self.bits_overrides:
            r += ", overrides=" + repr(self.bits_overrides)
        return r + ")"

