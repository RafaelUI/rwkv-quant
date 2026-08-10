"""
Функции fake-квантования весов RWKV-7: обычное symmetric per-channel RTN,
percentile-clipped вариант, и SpQR-style sparse outlier extraction.

Ключевая находка (см. README проекта): clipping спасает группы, где выброс --
это "мусорное" значение на фоне плотного нормального кластера (напр. r_k),
но ВРЕДИТ полноранговым dense-матрицам (proj/cmix), где "хвост" распределения
несёт реальный обученный сигнал. Для dense-групп нужен SpQR-style подход:
сохранить выбросы точно (разреженно, в bf16), а не резать/искажать их.
"""
import os

import torch
import torch.nn.functional as F

from .group_config import QuantConfig
from . import groupwise as _gw


def fake_quantize_sparse_outlier(w: torch.Tensor, bits: int, outlier_frac: float) -> torch.Tensor:
    """SpQR-style: держим top outlier_frac-долю значений КАЖДОЙ строки в exact bf16
    (разреженно, поэлементно), остальное квантуем с чистой шкалой (без искажения
    выбросами). outlier_frac=0.01 -> топ-1% значений строки остаются точными."""
    orig_dtype = w.dtype
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
    qv = torch.clamp(torch.round(w_dense / scale), -qmax - 1, qmax)
    deq = qv * scale
    result = torch.where(outlier_mask, w32, deq)
    return result.to(orig_dtype)


def fake_quantize(w: torch.Tensor, bits: int, per_channel: bool = True, clip_percentile=None) -> torch.Tensor:
    if bits >= 16:
        return w
    orig_dtype = w.dtype
    w32 = w.float()
    qmax = 2 ** (bits - 1) - 1
    if clip_percentile is not None and per_channel and w32.dim() >= 2:
        flat = w32.abs().reshape(w32.shape[0], -1)
        clip_val = torch.quantile(flat, clip_percentile / 100, dim=1, keepdim=True).clamp_min(1e-8)
        shape = [w32.shape[0]] + [1] * (w32.dim() - 1)
        clip_val = clip_val.view(*shape)
    elif clip_percentile is not None:
        clip_val = torch.quantile(w32.abs().reshape(-1), clip_percentile / 100).clamp_min(1e-8)
    elif per_channel and w32.dim() >= 2:
        clip_val = w32.abs().amax(dim=tuple(range(1, w32.dim())), keepdim=True).clamp_min(1e-8)
    else:
        clip_val = w32.abs().amax().clamp_min(1e-8)
    scale = clip_val / qmax
    wc = torch.clamp(w32, -clip_val, clip_val)
    qv = torch.clamp(torch.round(wc / scale), -qmax - 1, qmax)
    return (qv * scale).to(orig_dtype)


# ---------------- кеш квантованных весов ----------------
#
# Блочные схемы с грид-поиском стоят на два порядка дороже per-row RTN, а
# q() зовётся на КАЖДЫЙ тензор КАЖДОГО батча. Без кеша ablation на
# groupwise превращается из минут в часы.
#
# Ключ включает id(w), и вместе с ним в записи лежит СИЛЬНАЯ ссылка на
# сам w. Это не расточительность: без неё освобождённый тензор мог бы
# отдать свой адрес новому, и кеш молча вернул бы чужие веса. Веса
# модели и так живут весь прогон, так что ссылка ничего не стоит.
#
# ПОЛИТИКА «ЛИБО ВЕСЬ РАБОЧИЙ НАБОР, ЛИБО НИЧЕГО» -- вывод из замера, а
# не из вкуса. tests/probe_ppl_memory.py и /tmp-бенч на 1.5B (M4, 16 ГБ,
# чередование в одном процессе, закон 1):
#
#   ИЗОЛИРОВАННО (одна группа cmix, рабочий набор влезает):
#     кеш ВЫКЛ  69.3 / 68.7 с      кеш ВКЛ  33.2 / 33.2 с   -> 2.08x
#     ppl бит-в-бит, своп не двинулся
#
#   КОМПОЗИТ (квантованы ВСЕ группы, рабочий набор ~2.8 ГБ, не влезает):
#     кеш ВЫКЛ   90 с, пик свопа +0 МБ
#     кеш 1536 МБ  119 с, пик свопа +4424 МБ
#
# То есть при нехватке места частичный кеш проигрывает ОТСУТСТВИЮ кеша по
# ОБЕИМ осям сразу: он и медленнее (промахи + давление на память), и
# устраивает своп, который по закону 11 делает замер недействительным.
# Поэтому вытеснения здесь нет вовсе: как только рабочий набор перестаёт
# помещаться в бюджет, кеш ОТКЛЮЧАЕТСЯ до конца замера и освобождает уже
# накопленное. Промежуточного режима не предусмотрено намеренно -- он
# измеренно хуже обоих крайних.
#
# Бюджет выводится из СВОБОДНОЙ памяти в момент старта замера, а не из
# константы: 1536 МБ выглядели скромно и всё равно утащили машину в своп,
# потому что пол замера уже занимал 76% памяти.
_CACHE = {}
_CACHE_ON = False
_CACHE_BYTES = 0
_CACHE_BUDGET = 0
_CACHE_GAVE_UP = False
# доля СВОБОДНОЙ памяти, которую позволено занять кешу; переопределяется
# RWKVQ_CACHE_FRACTION, жёсткий потолок -- RWKVQ_CACHE_MAX_MB
_CACHE_FRACTION = float(os.environ.get("RWKVQ_CACHE_FRACTION", "0.25"))
_CACHE_MAX = int(os.environ.get("RWKVQ_CACHE_MAX_MB", "2048")) << 20


def available_bytes():
    """Свободная память по vm_stat: free + inactive + speculative.

    Не RSS и не ps: для unified memory они бесполезны (закон 11).
    inactive засчитывается, потому что ядро отдаёт эти страницы без
    записи в своп. При невозможности определить -- ноль, и кеш просто не
    включится: лучше медленно, чем в свопе.
    """
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = 16384
        first = out.splitlines()[0]
        if "page size of" in first:
            page = int(first.split("page size of")[1].split("bytes")[0].strip())
        n = {}
        for line in out.splitlines()[1:]:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            n[k.strip()] = int(v.strip().rstrip("."))
        return page * (n.get("Pages free", 0) + n.get("Pages inactive", 0)
                       + n.get("Pages speculative", 0))
    except Exception:                                        # noqa: BLE001
        return 0


def cache_begin(budget_bytes=None):
    """Включить кеш на время одного замера (см. ablation.perplexity)."""
    global _CACHE_ON, _CACHE_BYTES, _CACHE_BUDGET, _CACHE_GAVE_UP
    _CACHE.clear()
    _CACHE_BYTES, _CACHE_GAVE_UP = 0, False
    _CACHE_BUDGET = (budget_bytes if budget_bytes is not None
                     else min(int(available_bytes() * _CACHE_FRACTION), _CACHE_MAX))
    _CACHE_ON = _CACHE_BUDGET > 0


def cache_end():
    """Выключить и освободить."""
    global _CACHE_ON, _CACHE_BYTES
    _CACHE_ON = False
    _CACHE_BYTES = 0
    _CACHE.clear()


def cache_stats():
    return {"entries": len(_CACHE), "bytes": _CACHE_BYTES,
            "budget": _CACHE_BUDGET, "gave_up": _CACHE_GAVE_UP}


def _cache_put(key, w, out):
    """Положить в кеш либо, если бюджет исчерпан, сдаться до конца замера."""
    global _CACHE_BYTES, _CACHE_ON, _CACHE_GAVE_UP
    nbytes = out.numel() * out.element_size()
    if _CACHE_BYTES + nbytes > _CACHE_BUDGET:
        _CACHE_ON, _CACHE_GAVE_UP = False, True
        _CACHE.clear()
        _CACHE_BYTES = 0
        return
    _CACHE[key] = (w, out)
    _CACHE_BYTES += nbytes


def _gw_fake(w, group, cfg, key):
    """Блочная (group-wise) схема -- ТА ЖЕ диспетчеризация, что в
    formats.writer.quantize_tensor, ветка real_gw=False.

    Это и есть смысл функции: до 08.2026 калибровка меряла per-row RTN
    независимо от того, что стоит в group_scale, а деплоили groupwise
    sb6. `q(w, g, QuantConfig(proj=4))` и
    `q(w, g, QuantConfig(proj=4, group_scale={"proj": 32}))` возвращали
    ПОБИТОВО ОДНО И ТО ЖЕ, то есть критерий, по которому api.calibrate()
    выбирает битность, не имел отношения к схеме, попадающей в файл.
    Любая правка диспетчеризации здесь обязана повторяться в writer --
    гейт tests/test_calib_matches_writer.py требует совпадения выходов.
    """
    bits = cfg.bits[group]
    gs = cfg.group_scale[group]
    mode = cfg.group_scale_mode.get(group, "asym")
    sp = getattr(cfg, "act_stats_path", None)
    frac = cfg.outlier_fracs.get(group, 0.0)

    if mode.startswith("sym"):
        ex2 = _gw.get_ex2(sp, key, w) if mode.endswith("_aw") else None
        return _gw.groupwise_sym_fake_dequant(
            w, bits, gs=gs, sb=max(1, 256 // gs), ex2=ex2,
            search=not mode.endswith("_plain"), outlier_frac=frac)
    if mode == "mxfp4":
        return _gw.mxfp4_fake_dequant(w, gs, frac)
    if mode == "asym_sb6":
        return _gw.groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=6)
    if mode == "asym_sb6_search":
        return _gw.groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=-6)
    if mode == "asym_sb6_aw":
        return _gw.groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=-6,
                                          ex2=_gw.get_ex2(sp, key, w))
    return _gw.groupwise_fake_dequant(w, bits, gs)


def q(w, group, cfg: "QuantConfig", key: str = None):
    """Fake-квантование одного тензора по конфигу.

    key -- ключ state_dict. Нужен только AW-режимам (статистика активаций
    адресуется ключом); models/rwkv7_ref.py передаёт его там, где он
    известен. Без key AW-режим вырождается в свой _search-вариант -- ровно
    так же, как writer ведёт себя при отсутствующем файле статистики.
    """
    bits = cfg.bits[group]
    gs = cfg.group_scale.get(group)

    if gs and bits < 16:
        if w.dim() != 2:
            # writer на таком тензоре упал бы в reshape внутри блочной
            # функции. Молча уйти в per-row значило бы вернуть ровно тот
            # дефект, ради которого этот код и написан: измерить одну
            # схему, задеплоить другую.
            raise ValueError(
                f"group_scale={gs} задан для группы {group!r}, но тензор "
                f"{key or '<без ключа>'} имеет форму {tuple(w.shape)} -- "
                f"блочная схема определена только для 2-D. Уберите группу "
                f"из group_scale либо оставьте ей bits=16.")
        if not _CACHE_ON:
            return _gw_fake(w, group, cfg, key).to(w.dtype)
        ck = (id(w), group, bits, gs, cfg.group_scale_mode.get(group, "asym"),
              cfg.outlier_fracs.get(group, 0.0),
              getattr(cfg, "act_stats_path", None), key)
        hit = _CACHE.get(ck)
        if hit is not None:
            return hit[1]
        out = _gw_fake(w, group, cfg, key).to(w.dtype)
        _cache_put(ck, w, out)      # сильная ссылка на w -- см. у _CACHE
        return out

    if group in cfg.outlier_fracs and w.dim() >= 2:
        return fake_quantize_sparse_outlier(w, bits, cfg.outlier_fracs[group])
    return fake_quantize(w, bits, clip_percentile=cfg.clip_percentiles.get(group))
