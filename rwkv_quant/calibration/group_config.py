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
    def __init__(self, clip_percentiles=None, outlier_fracs=None, **bits_per_group):
        self.bits = {g: 16 for g in GROUPS}
        self.bits.update(bits_per_group)
        self.clip_percentiles = clip_percentiles or {}
        self.outlier_fracs = outlier_fracs or {}

    def __repr__(self):
        return "QuantConfig(" + ", ".join(f"{g}={self.bits[g]}" for g in GROUPS) + ")"

