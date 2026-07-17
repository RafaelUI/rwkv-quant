"""
Настоящее (не fake) квантование + сохранение в .rwkvq.

Осознанно отделено от calibration.fake_quant: там задача -- измерить
ppl-эффект (дёшево, дёргается тысячи раз при ablation), здесь -- один раз
произвести реальные упакованные коды для сохранения на диск.
"""
import torch

from ..calibration.group_config import QuantConfig
from ..calibration.outlier_scan import GROUP_KEY_PATTERNS
from .schema import QuantizedTensor, QuantizedCheckpoint, pack_int4


def _real_quantize(w: torch.Tensor, bits: int):
    """RTN per-row: возвращает (codes int8, scale fp16 [n_rows,1])."""
    w32 = w.float()
    qmax = 2 ** (bits - 1) - 1
    if w32.dim() >= 2:
        amax = w32.abs().amax(dim=tuple(range(1, w32.dim())), keepdim=True).clamp_min(1e-8)
    else:
        amax = w32.abs().amax().clamp_min(1e-8)
    scale = (amax / qmax)
    codes = torch.clamp(torch.round(w32 / scale), -qmax - 1, qmax).to(torch.int8)
    return codes, scale.to(torch.float16)


def _real_quantize_sparse_outlier(w: torch.Tensor, bits: int, outlier_frac: float):
    """SpQR-style: outlier-позиции исключаются из scale и codes (получают code=0),
    их точные значения + (row,col) индексы хранятся отдельно, разреженно."""
    w32 = w.float()
    n_cols = w32.shape[1]
    k = max(1, int(round(n_cols * outlier_frac)))
    abs_w = w32.abs()
    kth_val = torch.topk(abs_w, k, dim=1, largest=True).values[:, -1:].clamp_min(1e-8)
    outlier_mask = abs_w >= kth_val

    w_dense = torch.where(outlier_mask, torch.zeros_like(w32), w32)
    qmax = 2 ** (bits - 1) - 1
    amax = w_dense.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    codes = torch.clamp(torch.round(w_dense / scale), -qmax - 1, qmax).to(torch.int8)
    codes = torch.where(outlier_mask, torch.zeros_like(codes), codes)

    rows, cols = torch.where(outlier_mask)
    outlier_indices = torch.stack([rows, cols], dim=1).to(torch.int32)
    outlier_values = w32[rows, cols].to(torch.bfloat16)

    return codes, scale.to(torch.float16), outlier_indices, outlier_values


def _match_group(key: str):
    for group, pats in GROUP_KEY_PATTERNS.items():
        if any(key.endswith(pat) or pat in key for pat in pats):
            return group
    return None


# models/rwkv7_ref.py НИКОГДА не квантует эти bias-термы LoRA-веток (w0/a0/v0
# для world naming, *_lora_B.bias для custom) -- в forward они используются
# raw, не через q(...) (см. rwkv7_ref.py: F.linear(..., t.w_lora_B_b) без
# обёртки). Если квантовать их здесь вслепую по паттерну группы, реальная
# упаковка расходится с тем, что calibrate()/fake_quant вообще оценивали --
# бага была обнаружена эмпирически: w0 имеет форму (1,1,C), per-row RTN на
# ней даёт ОДНУ scale на все C каналов decay-gate'а, что напрямую портит
# рекуррентность на каждом токене каждого слоя (ppl 11.4 -> 248 на 1.5B
# при w_lora=INT4, входит в состав объяснения взрыва COMPRESSION).
_LORA_BIAS_SUFFIXES = (".w_lora_B.bias", ".a_lora_B.bias", ".v_lora_B.bias", ".w0", ".a0", ".v0")


def _make_qt(key, group, bits, shape, codes, scale, oi=None, ov=None):
    """bits <= 4 -> нибблы (codes_packed), иначе int8 codes as-is."""
    if bits <= 4:
        return QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(shape),
                               codes_packed=pack_int4(codes), scale=scale,
                               outlier_indices=oi, outlier_values=ov)
    return QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(shape),
                           codes=codes, scale=scale,
                           outlier_indices=oi, outlier_values=ov)


def quantize_tensor(key: str, w: torch.Tensor, cfg: QuantConfig) -> QuantizedTensor:
    group = _match_group(key)
    if group is None or w.dim() < 2 or key.endswith(_LORA_BIAS_SUFFIXES):
        return QuantizedTensor(key=key, group=group or "other", bits=16, shape=tuple(w.shape),
                                dense=w.to(torch.bfloat16))

    bits = cfg.bits[group]
    for pat, b in getattr(cfg, "bits_overrides", {}).items():
        if pat in key:
            bits = b
            break
    if bits >= 16:
        return QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape),
                                dense=w.to(torch.bfloat16))

    if group in cfg.outlier_fracs:
        codes, scale, oi, ov = _real_quantize_sparse_outlier(w, bits, cfg.outlier_fracs[group])
        return _make_qt(key, group, bits, w.shape, codes, scale, oi, ov)

    # clip_percentiles игнорируется здесь по конструкции: percentile-clipping
    # хорош для измерения ppl (fake_quant), но для реальной упаковки нужен
    # либо SpQR (outlier_fracs), либо обычный RTN -- см. README про то, почему
    # clipping вредит dense-группам.
    codes, scale = _real_quantize(w, bits)
    return _make_qt(key, group, bits, w.shape, codes, scale)


def save(state_dict: dict, config: QuantConfig, output_path: str,
         naming: str, n_layer: int, n_embd: int, head_size: int, vocab_size: int):
    tensors = {}
    for key, w in state_dict.items():
        tensors[key] = quantize_tensor(key, w, config)

    ckpt = QuantizedCheckpoint(
        naming=naming, n_layer=n_layer, n_embd=n_embd, head_size=head_size,
        vocab_size=vocab_size, tensors=tensors, config_repr=repr(config),
    )
    torch.save(ckpt, output_path)
    return ckpt
