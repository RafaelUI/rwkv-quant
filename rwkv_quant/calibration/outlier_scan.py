"""
Профилирование per-channel выбросов в весах RWKV-7.

Находка, которая мотивировала SpQR-style квантование в fake_quant.py:
в r_k/k_k/k_a и даже в полноранговых proj/cmix матрицах встречаются
поканальные (per-row) отношения max/mean в диапазоне 40-96x. При
symmetric per-channel квантовании scale = max/qmax, поэтому один такой
выброс уничтожает разрешение для всех остальных значений строки.

GROUP_KEY_PATTERNS покрывает обе схемы именования чекпоинтов (custom и
official "world") -- см. models/naming.py.
"""
import torch

GROUP_KEY_PATTERNS = {
    "proj": ["r_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
             "att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight"],
    "w_lora": ["w_lora_A", "w_lora_B", ".w0", ".w1", ".w2"],
    "a_lora": ["a_lora_A", "a_lora_B", ".a0", ".a1", ".a2"],
    "v_lora": ["v_lora_A", "v_lora_B", ".v0", ".v1", ".v2"],
    "g_lora": ["g_lora_A", "g_lora_B", ".g1", ".g2"],
    "small": ["k_k", "k_a", "r_k"],
    "cmix": ["cmix.key.weight", "cmix.value.weight", "ffn.key.weight", "ffn.value.weight"],
    "emb_head": ["emb.weight", "head.weight"],
}

# Bias-термы LoRA-веток, которые НЕ квантуются никогда и ни на одном пути:
# models/rwkv7_ref.py использует их в forward сырыми (F.linear(..., t.w_lora_B_b)
# без обёртки q()), а formats/writer.quantize_tensor отдаёт их плотными bf16.
# Причина не стилистическая: w0 имеет форму (1,1,C), и per-row RTN даёт ОДНУ
# scale на все C каналов decay-gate'а -- это напрямую портит рекуррентность на
# каждом токене каждого слоя (ppl 11.4 -> 248 на 1.5B при w_lora=INT4).
#
# Таблица живёт здесь, а не в writer, потому что на неё смотрят ОБА слоя:
# writer -- чтобы не квантовать, calibration/schema_space -- чтобы не считать
# эти тензоры при выводе применимости схем. Пока она была только в writer,
# schema_space видел у w_lora трёхмерный w0 и делал вывод, что блочные схемы
# группе недоступны, -- при том что квантуются там только двумерные w1/w2.
LORA_BIAS_SUFFIXES = (".w_lora_B.bias", ".a_lora_B.bias", ".v_lora_B.bias",
                      ".w0", ".a0", ".v0")


def scan_channel_outliers(state_dict, tensor_names_by_layer, label: str = "", top_n: int = 5):
    """
    tensor_names_by_layer: список (layer_idx, key) пар для одного типа тензора
    (напр. все blocks.N.att.receptance.weight по слоям).
    Возвращает топ-N слоёв по худшему поканальному отношению max/mean,
    отсортированные по убыванию.
    """
    worst = []
    for layer_idx, key in tensor_names_by_layer:
        t = state_dict[key].float()
        amax_per_row = t.abs().amax(dim=1)
        amean_per_row = t.abs().mean(dim=1).clamp_min(1e-8)
        ratio = (amax_per_row / amean_per_row).max().item()
        worst.append((layer_idx, ratio, amax_per_row.max().item()))
    worst.sort(key=lambda x: -x[1])
    return worst[:top_n]


def group_param_counts(state_dict, groups=None):
    """Считает количество параметров и типичное число входных колонок
    (для оценки бит на индекс в SpQR-надстройке) по каждой квантуемой группе."""
    from .group_config import GROUPS
    groups = groups or GROUPS
    counts = {g: 0 for g in groups}
    incols = {g: [] for g in groups}
    other = 0
    for key, tensor in state_dict.items():
        n = tensor.numel()
        matched = False
        for g, pats in GROUP_KEY_PATTERNS.items():
            if g not in groups:
                continue
            if any(key.endswith(pat) or pat in key for pat in pats):
                counts[g] += n
                if tensor.dim() >= 2:
                    incols[g].append(tensor.shape[1])
                matched = True
                break
        if not matched:
            other += n
    return counts, incols, other
