"""
Настоящее (не fake) квантование + сохранение в .rwkvq.

Осознанно отделено от calibration.fake_quant: там задача -- измерить
ppl-эффект (дёшево, дёргается тысячи раз при ablation), здесь -- один раз
произвести реальные упакованные коды для сохранения на диск.
"""
import json
import os
import warnings

import torch

from ..calibration.group_config import QuantConfig
from ..calibration.outlier_scan import GROUP_KEY_PATTERNS, LORA_BIAS_SUFFIXES
from ..calibration import groupwise as _gw
from ..models.naming import detect_naming
from . import codec
from .schema import (QuantizedTensor, QuantizedCheckpoint, pack_int4,
                     pack6, pack_nib_block, pack_bitplane)

# Блочные fake-quant схемы ПЕРЕЕХАЛИ в calibration/groupwise.py -- см.
# докстринг того файла: пока они жили здесь, слой калибровки не мог их
# позвать (writer импортирует calibration, обратное направление -- цикл),
# и `calibration.fake_quant.q()` не знала про group_scale вообще.
# Здесь -- ИМЕНА-ПСЕВДОНИМЫ, а не копии: `is`-тождество, поэтому
# разойтись физически нечему. Прежние имена сохранены, потому что на них
# ссылаются NEXT_SESSION.md и скрипты в tests/.
_groupwise_fake_dequant = _gw.groupwise_fake_dequant
_groupwise_sym_fake_dequant = _gw.groupwise_sym_fake_dequant
_mxfp4_fake_dequant = _gw.mxfp4_fake_dequant
_load_act_stats = _gw.load_act_stats
_get_ex2 = _gw.get_ex2

# поля QuantizedTensor, которые едут на диск отдельными буферами; всё
# остальное (bits/shape/group/gw_mode/gw_gs/gw_sb) -- скаляры и живут в
# манифесте
TENSOR_FIELDS = (
    "codes", "codes_packed", "scale", "dense",
    "outlier_indices", "outlier_values",
    "gw_d", "gw_dm", "gw_qsqm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
)


def config_to_json(cfg: QuantConfig) -> dict:
    """QuantConfig -> структура, из которой его можно собрать обратно.

    Раньше в файл шёл только repr(), а он ЛОССИ: QuantConfig.__repr__
    печатает битность групп и overrides, но молча теряет group_scale,
    group_scale_mode, clip_percentiles, outlier_fracs и act_stats_path --
    то есть ровно то, чем REDUCTION отличается от COMPRESSION при
    одинаковой битности. Воспроизвести квантование по такому файлу было
    нельзя. repr() остаётся рядом как человекочитаемая строка."""
    return {
        "bits": dict(cfg.bits),
        "clip_percentiles": dict(cfg.clip_percentiles),
        "outlier_fracs": dict(cfg.outlier_fracs),
        "bits_overrides": dict(cfg.bits_overrides),
        "group_scale": dict(cfg.group_scale),
        "group_scale_mode": dict(cfg.group_scale_mode),
        "act_stats_path": cfg.act_stats_path,
    }


def _n_blocks(qt) -> int:
    """Число блоков, которое РЕАЛЬНО покрывают квальные буферы.

    Не выводится делением нацело: у asym блоков ceil(IN/gs), потому что
    хвостовой неполный блок получает свой scale (`att.w1` [2048, 96] при
    gs=64 -- два блока, а IN//gs = 1). Берём из формы буфера, чтобы
    манифест утверждал факт, а не гипотезу."""
    if qt.gw_mode == "sb6":
        return int(qt.gw_qsqm.shape[-2]) * int(qt.gw_sb)
    if qt.gw_mode == "asym":
        return int(qt.gw_scale.shape[-1])
    return 0


def save_rwkvq(ckpt: QuantizedCheckpoint, output_path: str,
               config: QuantConfig = None, tokenizer: str = None):
    """QuantizedCheckpoint -> .rwkvq в контейнере safetensors.

    Плоские имена "<ключ>::<поле>", метаданные -- одним JSON в
    "__metadata__"["rwkvq"]. Чем это лучше прежнего torch.save:

      - читается без torch И без установленного пакета rwkv_quant
        (pickle писал полные имена классов, поэтому файл был привязан к
        версии пакета) -- см. codec.open_rwkvq, полная читалка на numpy;
      - mmap и потензорное чтение вместо «всё в анонимную память»;
      - нет исполнения произвольного кода при загрузке
        (torch.load(weights_only=False) буквально исполняет файл);
      - zero-copy в MLX/Metal и читаемость из Rust/C++/Swift.

    Прежние файлы продолжают читаться: reader.load_raw различает
    контейнеры по первым байтам (torch.save пишет zip PK\\x03\\x04).
    """
    tensors, manifest_t = {}, {}
    for key, qt in ckpt.tensors.items():
        fields = []
        for f in TENSOR_FIELDS:
            v = getattr(qt, f, None)
            if v is None or v.numel() == 0:
                continue
            # safetensors требует непрерывного буфера; для dense это ещё
            # и обрывает связь с mmap'нутым исходником
            tensors[f"{key}::{f}"] = v.detach().contiguous()
            fields.append(f)
        manifest_t[key] = {
            "kind": "dense" if qt.bits >= 16 else (qt.gw_mode or "rtn"),
            "shape": [int(x) for x in qt.shape],
            "bits": int(qt.bits),
            "group": qt.group or "other",
            "gw_gs": int(qt.gw_gs),
            "gw_sb": int(qt.gw_sb),
            "n_blocks": _n_blocks(qt),
            # ориентация: см. codec.is_transposed. Пишем факт, а не
            # оставляем потребителю таблицу имён и надежду на внимание
            "transposed": (ckpt.naming == "world"
                           and codec.is_raw_lora_world(key)),
            "fields": fields,
        }

    if config is None:
        config = getattr(ckpt, "config", None)
    manifest = {
        "format": codec.FORMAT,
        "format_version": codec.FORMAT_VERSION,
        "naming": ckpt.naming, "n_layer": int(ckpt.n_layer),
        "n_embd": int(ckpt.n_embd), "head_size": int(ckpt.head_size),
        "vocab_size": int(ckpt.vocab_size),
        "config_repr": ckpt.config_repr,
        "config": config_to_json(config) if config is not None else None,
        "tokenizer": tokenizer if tokenizer is not None
        else getattr(ckpt, "tokenizer", None),
        "tensors": manifest_t,
    }
    from safetensors.torch import save_file
    save_file(tensors, output_path, metadata={"rwkvq": json.dumps(manifest)})
    return ckpt


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


def _weighted_rtn_rows(w32, bits, ex2, chunk=2048):
    """Per-row RTN с activation-aware выбором scale (imatrix-стиль):
    для каждой строки грид по s=amax/qmax*f, f in [0.5..1.05], критерий
    sum_j ex2_j*(w_j - q_j*s)^2; затем s уточняется взвешенным LS по
    выбранным кодам: s* = sum(ex2*w*q)/sum(ex2*q*q). Возвращает
    (codes int8, scale fp32 [rows,1]). ex2 нормируется -> численно
    безопасно и не влияет на argmin."""
    qmax = 2 ** (bits - 1) - 1
    ex2 = (ex2 / ex2.mean().clamp_min(1e-12)).view(1, -1)
    fs = torch.linspace(0.5, 1.05, 23)
    out_codes, out_scale = [], []
    for r0 in range(0, w32.shape[0], chunk):
        wc = w32[r0:r0 + chunk]                                  # [R, IN]
        amax = wc.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        base = (amax / qmax)                                     # [R, 1]
        best_err = torch.full((wc.shape[0],), float("inf"))
        best_q = torch.zeros_like(wc, dtype=torch.int8)
        for f in fs:
            sc = base * f
            qc = torch.clamp(torch.round(wc / sc), -qmax - 1, qmax)
            err = (ex2 * (wc - qc * sc) ** 2).sum(dim=1)
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_q[m] = qc[m].to(torch.int8)
        qf = best_q.float()
        num = (ex2 * wc * qf).sum(dim=1, keepdim=True)
        den = (ex2 * qf * qf).sum(dim=1, keepdim=True).clamp_min(1e-12)
        out_codes.append(best_q)
        out_scale.append(num / den)
    return torch.cat(out_codes), torch.cat(out_scale)


def _weighted_quantize(w, bits, ex2, outlier_frac=0.0):
    """Activation-aware вариант _real_quantize(_sparse_outlier): выбросы
    отбираются по ex2*w^2 (цена ошибки), не по |w|; scale -- взвешенный
    грид (см. _weighted_rtn_rows). Формат вывода идентичен обычному."""
    w32 = w.float()
    if outlier_frac > 0:
        n_cols = w32.shape[1]
        k = max(1, int(round(n_cols * outlier_frac)))
        cost = w32 * w32 * ex2.view(1, -1)
        kth = torch.topk(cost, k, dim=1, largest=True).values[:, -1:].clamp_min(1e-20)
        mask = cost >= kth
        w_dense = torch.where(mask, torch.zeros_like(w32), w32)
        codes, scale = _weighted_rtn_rows(w_dense, bits, ex2)
        codes = torch.where(mask, torch.zeros_like(codes), codes)
        rows, cols = torch.where(mask)
        oi = torch.stack([rows, cols], dim=1).to(torch.int32)
        ov = w32[rows, cols].to(torch.bfloat16)
        return codes, scale.to(torch.float16), oi, ov
    codes, scale = _weighted_rtn_rows(w32, bits, ex2)
    return codes, scale.to(torch.float16), None, None


def _make_qt_gw_sb6(key, group, bits, w, gs, ex2, search=True):
    """Реальный формат v2 (sb6): нибблы блок-локального split + qh/qh2-
    битплоскости (bits=5: qh -- бит4; bits=6: qh + qh2 -- биты 4 и 5) +
    d/dm fp16 + qs/qm по 6 бит (qm со сдвигом +31). Дискретизация идентична
    fake-пути (return_parts) -- бит-точность проверяется тестом.
    search=False воспроизводит fake-режим "asym_sb6" (sb_bits=6, БЕЗ
    грид-поиска) -- нужен, т.к. на bits=6 AW/поиск не universally помогают
    (сессия 19.07-5: AW вредит для proj на 6 битах), а REDUCTION v2 держит
    proj именно на "asym_sb6" (search=False, ex2=None)."""
    assert bits in (4, 5, 6)
    OUT, IN = w.shape
    assert IN % gs == 0, f"{key}: IN={IN} не кратно gs={gs}"
    assert (IN // gs) % 8 == 0, f"{key}: NB={IN//gs} не кратно sb=8"
    parts = _groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=(-6 if search else 6),
                                    ex2=ex2, return_parts=True)
    q = parts["q"]                                   # uint8 0..2^bits-1
    qs, qm = parts["qs"], parts["qm"]                # [OUT, NB]
    qsqm = torch.cat([pack6(qs.view(OUT, -1, 8)),
                      pack6((qm.to(torch.int16) + 31).to(torch.uint8).view(OUT, -1, 8))],
                     dim=-1)                          # [OUT, NSB, 12]
    # bits=4: q уже 0..15, обе плоскости None. bits=5: +qh (бит4). bits=6:
    # +qh И qh2 (биты 4 и 5) -- каждый бит своей независимой плоскостью,
    # pack_bitplane переиспользуется без изменений (см. int5-прецедент).
    qh = qh2 = None
    if bits >= 5:
        qh = pack_bitplane(((q >> 4) & 1).to(torch.uint8).contiguous())
    if bits >= 6:
        qh2 = pack_bitplane(((q >> 5) & 1).to(torch.uint8).contiguous())
    q = q & 0xF
    return QuantizedTensor(
        key=key, group=group, bits=bits, shape=(OUT, IN),
        codes_packed=pack_nib_block(q, gs),
        gw_mode="sb6", gw_gs=gs, gw_sb=8,
        gw_d=parts["d"], gw_dm=parts["dm"], gw_qsqm=qsqm, gw_qh=qh, gw_qh2=qh2)


def _make_qt_gw_asym(key, group, bits, w, gs):
    """Реальный gw-asym (LoRA @6, gw64): int8-контейнер кодов (unsigned
    0..2^bits-1) + fp32 scale/min на блок -- бит-в-бит с fake-путём asym
    (там roundtrip'ов нет). Размер не жмём: группа крошечная (~25M парам)."""
    OUT, IN = w.shape
    parts = _groupwise_fake_dequant(w, bits, gs, return_parts=True)
    return QuantizedTensor(
        key=key, group=group, bits=bits, shape=(OUT, IN),
        codes=parts["q"],                             # uint8 as-is
        gw_mode="asym", gw_gs=gs,
        gw_scale=parts["scale"].squeeze(-1).float(),  # [OUT, NBpad]
        gw_min=parts["mn"].squeeze(-1).float())


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
# Таблица переехала в calibration/outlier_scan.py -- на неё смотрит и
# schema_space при выводе применимости схем (мотивация -- там же).
_LORA_BIAS_SUFFIXES = LORA_BIAS_SUFFIXES


def _make_qt(key, group, bits, shape, codes, scale, oi=None, ov=None):
    """bits <= 4 -> нибблы (codes_packed), иначе int8 codes as-is."""
    if bits <= 4:
        return QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(shape),
                               codes_packed=pack_int4(codes), scale=scale,
                               outlier_indices=oi, outlier_values=ov)
    return QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(shape),
                           codes=codes, scale=scale,
                           outlier_indices=oi, outlier_values=ov)


def quantize_tensor(key: str, w: torch.Tensor, cfg: QuantConfig,
                    real_gw: bool = False) -> QuantizedTensor:
    group = _match_group(key)
    if group is None or w.dim() < 2 or key.endswith(_LORA_BIAS_SUFFIXES):
        return QuantizedTensor(key=key, group=group or "other", bits=16, shape=tuple(w.shape),
                                dense=w.to(torch.bfloat16).clone().contiguous())

    bits = cfg.bits[group]
    for pat, b in getattr(cfg, "bits_overrides", {}).items():
        if pat in key:
            bits = b
            break
    sp = getattr(cfg, "act_stats_path", None)
    # gw-ветка РАНЬШЕ act_stats: иначе группа с group_scale и статистикой
    # ушла бы в per-row-AW и до блочного пути не дошла. AW внутри gw --
    # через режим asym_sb6_aw (ex2 в критерий поиска/LS).
    gs = getattr(cfg, "group_scale", {}).get(group)
    if gs and bits < 16:
        mode = getattr(cfg, "group_scale_mode", {}).get(group, "asym")
        if real_gw:
            # реальная упаковка формата v2 вместо dense fake-dequant
            if mode in ("asym_sb6", "asym_sb6_search", "asym_sb6_aw") and bits in (4, 5, 6):
                ex2 = _get_ex2(sp, key, w) if mode == "asym_sb6_aw" else None
                return _make_qt_gw_sb6(key, group, bits, w, gs, ex2,
                                       search=(mode != "asym_sb6"))
            if mode == "asym" and 5 <= bits <= 8:
                return _make_qt_gw_asym(key, group, bits, w, gs)
            raise NotImplementedError(f"real_gw: mode={mode} bits={bits} ({key})")
        if mode.startswith("sym"):
            # Q6_K-подобная раскладка (см. _groupwise_sym_fake_dequant).
            # gs здесь -- размер блока (16 у Q6_K против 32 у нашего sb6),
            # sb фиксирован так, чтобы суперблок оставался 256 весов.
            # "_aw" подаёт ex2 в критерий поиска, "_plain" убирает поиск.
            ex2 = _get_ex2(sp, key, w) if mode.endswith("_aw") else None
            deq = _groupwise_sym_fake_dequant(
                w, bits, gs=gs, sb=max(1, 256 // gs), ex2=ex2,
                search=not mode.endswith("_plain"),
                outlier_frac=cfg.outlier_fracs.get(group, 0.0))
        elif mode == "mxfp4":
            deq = _mxfp4_fake_dequant(w, gs, cfg.outlier_fracs.get(group, 0.0))
        elif mode == "asym_sb6":
            deq = _groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=6)
        elif mode == "asym_sb6_search":
            deq = _groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=-6)
        elif mode == "asym_sb6_aw":
            deq = _groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=-6,
                                          ex2=_get_ex2(sp, key, w))
        else:
            deq = _groupwise_fake_dequant(w, bits, gs)
        return QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape),
                                dense=deq.to(torch.bfloat16))
    if sp and bits < 16:
        ex2 = _get_ex2(sp, key, w)
        if ex2 is not None:
            frac = cfg.outlier_fracs.get(group, 0.0)
            codes, scale, oi, ov = _weighted_quantize(w, bits, ex2, frac)
            return _make_qt(key, group, bits, w.shape, codes, scale, oi, ov)
        # нет статистики (emb: вход -- индексы токенов; LoRA не писали) или
        # она от другой модели -- проваливаемся в обычный путь ниже
    if bits >= 16:
        return QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape),
                                dense=w.to(torch.bfloat16).clone().contiguous())

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
         naming: str, n_layer: int, n_embd: int, head_size: int, vocab_size: int,
         real_gw: bool = True, tokenizer: str = None):
    """Квантовать state_dict и записать .rwkvq.

    real_gw=True (по умолчанию) -- РЕАЛЬНАЯ упаковка: группы с group_scale
    пакуются в формат sb6 и файл действительно сжимается.

    real_gw=False -- fake-quant: веса прогоняются через квантование и обратно,
    но сохраняются плотными в bf16. Это режим ИЗМЕРЕНИЯ (оценить деградацию
    ppl, не трогая кернели), а не выдачи артефакта.

    Раньше здесь безусловно вызывался quantize_tensor без real_gw, то есть с
    дефолтом False, и публичный API физически не мог произвести деплоимый
    файл: quantize("model.pth", "model.rwkvq", preset="reduction") из README
    отдавал bf16 в контейнере .rwkvq того же размера, что исходник. Реальная
    упаковка при этом существовала и была бит-в-бит проверена, но дотянуться
    до неё можно было только вызовом quantize_tensor(..., real_gw=True)
    вручную -- что и делали скрипты в tests/.
    """
    tensors = {}
    for key, w in state_dict.items():
        tensors[key] = quantize_tensor(key, w, config, real_gw=real_gw)

    ckpt = QuantizedCheckpoint(
        naming=naming, n_layer=n_layer, n_embd=n_embd, head_size=head_size,
        vocab_size=vocab_size, tensors=tensors, config_repr=repr(config),
        config=config, tokenizer=tokenizer,
    )
    return save_rwkvq(ckpt, output_path)


def detect_meta(checkpoint_path: str, state_dict) -> dict:
    """naming / n_layer / n_embd / head_size / vocab_size прямо из ключей.

    Раньше эти пять чисел брались инстанцированием RWKV7Ref -- то есть
    ради метаданных в память поднималась ВСЯ модель в bf16 (5.9 ГБ на
    2.9B), а потом выбрасывалась. Здесь читаются только формы, и при
    mmap-загрузке тела тензоров вообще не трогаются.

    head_size берётся из r_k -- он имеет форму [n_head, head_size] в обеих
    схемах именования (в world это blocks.N.att.r_k, в custom --
    blocks.N.tmix.r_k), поэтому размер головы не приходится угадывать.
    """
    n_layer = 1 + max(int(k.split(".")[1]) for k in state_dict
                      if k.startswith("blocks."))
    emb = state_dict["emb.weight"]
    r_k = next(v for k, v in state_dict.items() if k.endswith("r_k"))
    return {
        "naming": detect_naming(checkpoint_path, state_dict),
        "n_layer": n_layer,
        "n_embd": int(emb.shape[1]),
        "vocab_size": int(emb.shape[0]),
        "head_size": int(r_k.shape[-1]),
    }


def quantize_file(checkpoint_path: str, output_path: str, config: QuantConfig,
                  real_gw: bool = True, verbose: bool = True,
                  tokenizer: str = None):
    """Потоковое квантование чекпоинта: .pth/.safetensors -> .rwkvq.

    Отличие от save(): state_dict не держится в памяти целиком. Тензоры
    читаются через mmap (страницы file-backed, ядро вытесняет их само,
    не через своп) и по одному заменяются квантованными, исходная ссылка
    сразу отпускается.

    Зачем: прежний путь на 2.9B давал пик 9-12 ГБ -- 5.9 ГБ исходного
    state_dict, поверх него растущий словарь результата (2.4 ГБ) и fp32-
    воркспейс грид-поиска (до 1 ГБ на cmix 10240x2560, там держится
    порядка десяти копий блока). На 16 ГБ это гарантированный своп, а
    своп во время квантования 2.9B -- десятки минут. Здесь пик равен
    результату плюс воркспейс одного тензора, около 2.5 ГБ.

    torch.load(mmap=True) требует zip-сериализации (все современные .pth)
    и map_location="cpu".
    """
    if checkpoint_path.endswith(".pth"):
        sd = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    else:
        from safetensors.torch import load_file
        path = checkpoint_path
        if os.path.isdir(path):
            path = os.path.join(path, "model.safetensors")
        sd = load_file(path)

    meta = detect_meta(checkpoint_path, sd)
    if verbose:
        print(f"{os.path.basename(checkpoint_path)}: {len(sd)} тензоров, "
              f"naming={meta['naming']} L={meta['n_layer']} "
              f"D={meta['n_embd']} H={meta['head_size']} "
              f"V={meta['vocab_size']}", flush=True)

    tensors = {}
    keys = list(sd.keys())
    for i, key in enumerate(keys):
        tensors[key] = quantize_tensor(key, sd[key], config, real_gw=real_gw)
        sd[key] = None            # отпускаем ссылку на mmap-вид немедленно
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(keys)}", flush=True)
    del sd

    ckpt = QuantizedCheckpoint(tensors=tensors, config_repr=repr(config),
                               config=config, tokenizer=tokenizer, **meta)
    save_rwkvq(ckpt, output_path)
    if verbose:
        print(f"-> {output_path} "
              f"({os.path.getsize(output_path)/1e6:.1f} МБ)", flush=True)
    return ckpt
