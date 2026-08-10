"""Блочные (group-wise) схемы квантования в fake-виде: квантуем и сразу
деквантуем обратно, веса остаются плотными.

ПОЧЕМУ ЭТОТ ФАЙЛ ЗДЕСЬ, А НЕ В formats/writer.py, ГДЕ ЖИЛ РАНЬШЕ.
Это функции ИЗМЕРЕНИЯ: они не производят ни одного упакованного байта,
их единственный потребитель по смыслу -- ablation/perplexity. Пока они
лежали в `formats/writer.py`, слой калибровки физически не мог их
позвать (writer импортирует calibration, обратное направление замкнуло
бы цикл), и `calibration.fake_quant.q()` -- функция, по которой
`api.calibrate()` выбирает битность -- не знала про `group_scale`
ВООБЩЕ. Побитово: `q(w, g, QuantConfig(proj=4))` и
`q(w, g, QuantConfig(proj=4, group_scale={"proj": 32}))` возвращали
одно и то же. То есть калибровка меряла per-row RTN, а деплоили
groupwise sb6 -- две разные функции, и решение о битности принималось
не по той из них.

`formats/writer.py` продолжает звать эти же объекты (импортом, не копией),
поэтому реальная упаковка не изменилась ни на бит -- гейт
tests/test_groupwise_move_parity.py это утверждает против замороженного
эталона.

Дискретизация здесь обязана совпадать с упаковщиками в writer:
`_make_qt_gw_sb6` берёт `return_parts=True` у ЭТОЙ функции и просто
раскладывает `q`/`qs`/`qm`/`d`/`dm` по битам. Любая правка математики
ниже меняет содержимое .rwkvq.
"""
import os
import warnings

import torch

__all__ = [
    "groupwise_fake_dequant", "groupwise_sym_fake_dequant",
    "mxfp4_fake_dequant", "load_act_stats", "get_ex2",
    "GW_MODES", "mode_needs_act_stats",
]

# Режимы group_scale_mode, которые понимает и калибровка, и writer.
# Ключ -> (нужна ли act-статистика, человекочитаемое описание).
GW_MODES = {
    "asym":             (False, "асимметричный RTN на блок, без суперблока"),
    "asym_sb6":         (False, "sb6 без грид-поиска (REDUCTION/proj)"),
    "asym_sb6_search":  (False, "sb6 + грид-поиск scale/min"),
    "asym_sb6_aw":      (True,  "sb6 + грид-поиск, взвешенный по E[x^2]"),
    "sym":              (False, "Q6_K-подобная симметричная раскладка + поиск"),
    "sym_plain":        (False, "то же без поиска"),
    "sym_aw":           (True,  "то же, поиск взвешен по E[x^2]"),
    "mxfp4":            (False, "OCP MXFP4 (E8M0 shared scale, элементы E2M1)"),
}


def mode_needs_act_stats(mode: str) -> bool:
    return GW_MODES.get(mode, (False, ""))[0]


# ---------------- статистика активаций (AW-режимы) ----------------

_ACT_STATS_CACHE = {}
_ACT_STATS_WARNED = set()
_EX2_MISMATCH_WARNED = set()


def load_act_stats(path):
    """Статистика активаций для AW-режимов; {} если файла нет.

    Пресеты указывают act_stats_path в /tmp (см. presets.py: "не переживает
    перезагрузку -- пересобрать перед использованием"), поэтому отсутствие
    файла -- НОРМАЛЬНАЯ ситуация, а не сбой. Раньше здесь был безусловный
    torch.load, и документированный в README вызов
        quantize("model.pth", "model.rwkvq", preset="reduction")
    падал с FileNotFoundError на первом же cmix-тензоре.

    Вырождение осмысленно: без статистики asym_sb6_aw ведёт себя как
    asym_sb6_search (ex2=None), то есть теряет activation-взвешивание, но
    остаётся корректным квантованием того же формата. Плюс статистика
    привязана к КОНКРЕТНОЙ модели -- молча применять стат от 1.5B к другому
    чекпоинту было бы хуже, чем не применять никакой.

    Кеш ключуется путём И mtime: раньше только путём, и пересборка
    статистики в том же процессе (обычный ноутбучный цикл) молча отдавала
    прежнюю.
    """
    if not path or not os.path.exists(path):
        if path and path not in _ACT_STATS_WARNED:
            _ACT_STATS_WARNED.add(path)
            warnings.warn(
                f"act_stats не найдены: {path}. AW-режимы (asym_sb6_aw) "
                "работают без activation-взвешивания. Пересоберите "
                "статистику или задайте config.act_stats_path=None, "
                "чтобы убрать это предупреждение.",
                RuntimeWarning, stacklevel=2)
        return {}
    key = (path, os.path.getmtime(path))
    if key not in _ACT_STATS_CACHE:
        _ACT_STATS_CACHE.clear()      # держим ровно одну статистику
        _ACT_STATS_CACHE[key] = torch.load(path)
    return _ACT_STATS_CACHE[key]


def get_ex2(path, key, w):
    """Статистика активаций для тензора -- или None, если её нет ИЛИ она
    от другой модели.

    Ключи чекпоинтов одинаковы у всех размеров RWKV-7, поэтому
    `stats.get(key)` радостно вернёт статистику 1.5B (2048 каналов) для
    0.1B (768) -- и это всплывёт лишь падением reshape где-то в
    groupwise_fake_dequant. Хуже того, при совпадении размерностей
    (например, две модели одного n_embd) не всплыло бы вовсе, и мы бы
    молча калибровали одну модель по активациям другой. Сверка длины --
    минимальная защита; полноценной была бы подпись чекпоинта в файле
    статистики.
    """
    if not path or key is None:
        return None
    ex2 = load_act_stats(path).get(key)
    if ex2 is None:
        return None
    if ex2.numel() != w.shape[-1]:
        tag = (path, w.shape[-1], int(ex2.numel()))
        if tag not in _EX2_MISMATCH_WARNED:
            _EX2_MISMATCH_WARNED.add(tag)
            warnings.warn(
                f"act_stats из {path} сняты для {ex2.numel()} входных "
                f"каналов, а тензор имеет {w.shape[-1]} -- статистика от "
                f"ДРУГОЙ модели, игнорируется. Пересоберите её на текущем "
                f"чекпоинте (tests/collect_act_stats.py).",
                RuntimeWarning, stacklevel=2)
        return None
    return ex2


# ---------------- асимметричная блочная схема (наш sb6) ----------------

def groupwise_fake_dequant(w: torch.Tensor, bits: int, gs: int,
                           sb: int = 0, sb_bits: int = 6, ex2=None,
                           return_parts: bool = False):
    """group-wise scale (ядро K-квантов): асимметричный RTN на блок
    из gs колонок (scale + min на блок), сразу деквантованный обратно.
    return_parts=True отдаёт сырые компоненты -- из них writer собирает
    реальную упаковку sb6, поэтому дискретизация здесь И в файле одна.
    Оверхед раскладки при gs=32, sb=8, sb_bits=6: 4 + (2*6)/32 + (2*16)/256
    = 4.5 бит/элемент."""
    w32 = w.float()
    OUT, IN = w32.shape
    pad = (-IN) % gs
    wp = torch.nn.functional.pad(w32, (0, pad)) if pad else w32
    wg = wp.view(OUT, wp.shape[1] // gs, gs)
    mn = wg.amin(dim=2, keepdim=True)
    mx = wg.amax(dim=2, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn) / qmax).clamp_min(1e-8)
    if ex2 is not None:
        # AW: вес колонки = E[x^2] её входного канала. Влияет только на
        # критерий поиска и LS, формат/раскладка не меняются.
        ev = ex2.float().clamp_min(1e-12)
        evp = torch.nn.functional.pad(ev, (0, pad)) if pad else ev
        evg = evp.view(1, wp.shape[1] // gs, gs)
    else:
        evg = None
    if sb and sb_bits < 0:  # sb_bits < 0: |sb_bits| + грид-поиск scale/min
        # 1) грид по фактору scale, 2) LS-дообводка (s, m) по выбранным
        # кодам (замкнутая форма на блок), 3) суперблочное квантование.
        # Аналог make_qkx2_quants (llama.cpp) поверх нашей раскладки.
        best_s, best_m, best_e = scale.clone(), mn.clone(), None
        for f in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05):
            sc = scale * f
            qq = torch.clamp(torch.round((wg - mn) / sc), 0, qmax)
            # LS: min_{s,m} sum (w - q*s - m)^2 на блок
            if evg is None:
                qm_ = qq.mean(dim=2, keepdim=True); wm_ = wg.mean(dim=2, keepdim=True)
                cov = ((qq - qm_) * (wg - wm_)).sum(dim=2, keepdim=True)
                var = ((qq - qm_) ** 2).sum(dim=2, keepdim=True).clamp_min(1e-12)
            else:  # взвешенная LS: те же формулы со средними по весам evg
                wsum = evg.sum(dim=2, keepdim=True)
                qm_ = (evg * qq).sum(dim=2, keepdim=True) / wsum
                wm_ = (evg * wg).sum(dim=2, keepdim=True) / wsum
                cov = (evg * (qq - qm_) * (wg - wm_)).sum(dim=2, keepdim=True)
                var = (evg * (qq - qm_) ** 2).sum(dim=2, keepdim=True).clamp_min(1e-12)
            s_ls = cov / var; m_ls = wm_ - s_ls * qm_
            s_ls = torch.where(s_ls > 1e-8, s_ls, sc)  # деградир. блоки
            q2 = torch.clamp(torch.round((wg - m_ls) / s_ls), 0, qmax)
            e2 = (q2 * s_ls + m_ls - wg) ** 2
            err = ((e2 if evg is None else evg * e2)).sum(dim=2, keepdim=True)
            if best_e is None:
                best_s, best_m, best_e = s_ls, m_ls, err
            else:
                b = err < best_e
                best_s = torch.where(b, s_ls, best_s)
                best_m = torch.where(b, m_ls, best_m)
                best_e = torch.minimum(best_e, err)
        scale, mn = best_s.clamp_min(1e-8), best_m
        sb_bits = -sb_bits
    if sb:
        # Q4_K-стиль: суперблок из sb блоков; scale/min блоков квантуются в
        # sb_bits против одной пары fp16 на суперблок (d, dm). Коды весов
        # выбираются ПОСЛЕ квантования scale/min (как в llama.cpp) -- ошибка
        # scale частично компенсируется выбором кодов.
        nb = scale.shape[1]
        pad_b = (-nb) % sb
        if pad_b:
            scale = torch.nn.functional.pad(scale, (0, 0, 0, pad_b))
            mn = torch.nn.functional.pad(mn, (0, 0, 0, pad_b))
        ssb = scale.view(OUT, -1, sb, 1); msb = mn.view(OUT, -1, sb, 1)
        smax = 2 ** sb_bits - 1                      # unsigned для scale>0
        # ВАЖНО (формат v2): d/dm проходят half-роундтрип ДО выбора qs/qm --
        # ровно эти half-значения лягут в файл, кернель восстановит
        # s = half(qs * float(d_half)) бит-в-бит с этим путём.
        d = (ssb.amax(dim=2, keepdim=True) / smax).clamp_min(1e-12).half().float()
        qs = torch.clamp(torch.round(ssb / d), 1, smax)
        scale_q = (qs * d).view(OUT, -1, 1)[:, :nb + pad_b][:, :nb]
        mmax = 2 ** (sb_bits - 1) - 1                # signed для min
        dm = (msb.abs().amax(dim=2, keepdim=True) / mmax).clamp_min(1e-12).half().float()
        qm = torch.clamp(torch.round(msb / dm), -mmax, mmax)
        mn_q = (qm * dm).view(OUT, -1, 1)[:, :nb + pad_b][:, :nb]
        # fp16-раунд-трип может занулить scale у (почти) константных
        # блоков (qs*d < 6e-8 -> half underflow) -> 0/0 = NaN в кодах.
        scale = scale_q.half().float().clamp_min(1e-8); mn = mn_q.half().float()
    qv = torch.clamp(torch.round((wg - mn) / scale), 0, qmax)
    deq = (qv * scale + mn).view(OUT, -1)[:, :IN]
    if not return_parts:
        return deq
    parts = {"q": qv.view(OUT, -1)[:, :IN].to(torch.uint8), "deq": deq,
             "scale": scale, "mn": mn}
    if sb:
        parts.update(qs=qs.view(OUT, -1, 1)[:, :nb].squeeze(-1).to(torch.uint8),
                     qm=qm.view(OUT, -1, 1)[:, :nb].squeeze(-1).to(torch.int8),
                     d=d.view(OUT, -1).half(), dm=dm.view(OUT, -1).half())
    return parts


# ---------------- симметричная блочная схема (Q6_K-подобная) ----------------

def groupwise_sym_fake_dequant(w: torch.Tensor, bits: int, gs: int = 16,
                               sb: int = 16, ex2=None, search: bool = True,
                               outlier_frac: float = 0.0) -> torch.Tensor:
    """ПРОТОТИП симметричного блочного кванта в раскладке block_q6_K
    (llama.cpp/ggml-common.h): scale на блок из gs весов БЕЗ min, сами
    scale квантуются в int8 против одной fp16-константы d на суперблок
    из sb блоков. Выход -- dense (fake-dequant).

    Зачем: наш asym_sb6 -- аналог block_q4_K (асимметрия, блок 32,
    6-битные scale/min), и мы применяем эту раскладку и на 4 битах, и на
    6. llama.cpp на шести битах структуру МЕНЯЕТ. Гипотеза: на 6 битах
    отдельный min не окупается -- распределение весов почти симметрично,
    а платим мы за него дважды (6 бит на min И урезанный до 6 бит scale),
    тогда как Q6_K тратит тот же бюджет на вдвое меньший блок с 8-битным
    scale.

    Бюджет: bits + 8/gs + 16/(gs*sb) бит на вес.
      bits=6, gs=16, sb=16 -> 6.5625 (ровно Q6_K)
      наш asym_sb6 при gs=32, sb=8 -> 6.5
    """
    w32 = w.float()
    OUT, IN = w32.shape
    omask = None
    if outlier_frac > 0.0:
        k = max(1, int(round(IN * outlier_frac)))
        kth = torch.topk(w32.abs(), k, dim=1).values[:, -1:].clamp_min(1e-20)
        omask = w32.abs() >= kth
        w_body = torch.where(omask, torch.zeros_like(w32), w32)
    else:
        w_body = w32

    pad = (-IN) % (gs * sb)
    wp = torch.nn.functional.pad(w_body, (0, pad)) if pad else w_body
    NB = wp.shape[1] // gs
    wg = wp.view(OUT, NB, gs)
    if ex2 is not None:
        ev = ex2.float().clamp_min(1e-12).view(1, -1)
        evp = torch.nn.functional.pad(ev, (0, pad)) if pad else ev
        evg = evp.view(1, NB, gs)
    else:
        evg = None

    qmax = 2 ** (bits - 1) - 1                    # 31 при bits=6
    qmin = -qmax - 1                              # -32
    amax = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    base = amax / qmax
    factors = (torch.linspace(0.7, 1.15, 19) if search
               else torch.tensor([1.0]))
    best_s, best_e = base.clone(), None
    for f in factors:
        s = base * f
        qv = torch.clamp(torch.round(wg / s), qmin, qmax)
        e = (wg - qv * s) ** 2
        err = (e if evg is None else evg * e).sum(dim=2, keepdim=True)
        if best_e is None:
            best_s, best_e = s, err
        else:
            b = err < best_e
            best_s = torch.where(b, s, best_s)
            best_e = torch.minimum(best_e, err)

    # суперблок: scale блоков -> int8 против общей fp16 d. d проходит
    # half-роундтрип ДО выбора кодов -- как в асимметричной ветке, чтобы
    # будущий кернель восстанавливал ровно эти числа.
    ssb = best_s.view(OUT, -1, sb, 1)
    d = (ssb.amax(dim=2, keepdim=True) / 127.0).clamp_min(1e-12).half().float()
    qs = torch.clamp(torch.round(ssb / d), -128, 127)
    scale_q = (qs * d).view(OUT, NB, 1).half().float()

    nz = scale_q.abs() > 0
    denom = torch.where(nz, scale_q, torch.ones_like(scale_q))
    qv = torch.clamp(torch.round(wg / denom), qmin, qmax)
    qv = torch.where(nz, qv, torch.zeros_like(qv))
    deq = (qv * scale_q).view(OUT, -1)[:, :IN]
    if omask is not None:
        deq = torch.where(omask, w32, deq)
    return deq


# ---------------- MXFP4 ----------------

_E2M1_GRID = None


def mxfp4_fake_dequant(w: torch.Tensor, gs: int,
                       outlier_frac: float = 0.0) -> torch.Tensor:
    """ПРОТОТИП MXFP4 (OCP MX): блок gs колонок, shared E8M0 scale (степень
    двойки), элементы FP4 E2M1 (+-{0,.5,1,1.5,2,3,4,6}). Экспонента блока
    выбирается перебором e0-1/e0/e0+1 по MSE блока. Эфф. битность:
    4 + 8/gs = 4.25 бит/элемент при gs=32 (против 5.0 у асимметричного
    gw32 c 2xfp16)."""
    global _E2M1_GRID
    if _E2M1_GRID is None:
        _E2M1_GRID = (torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.]),
                      torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.]))
    grid, mids = _E2M1_GRID
    w32 = w.float()
    OUT, IN = w32.shape
    omask = None
    if outlier_frac > 0.0:
        k = max(1, int(round(IN * outlier_frac)))
        kth = torch.topk(w32.abs(), k, dim=1, largest=True).values[:, -1:].clamp_min(1e-20)
        omask = w32.abs() >= kth
        w_body = torch.where(omask, torch.zeros_like(w32), w32)
    else:
        w_body = w32
    pad = (-IN) % gs
    wp = torch.nn.functional.pad(w_body, (0, pad)) if pad else w_body
    wg = wp.view(OUT, wp.shape[1] // gs, gs)
    amax = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    e0 = torch.ceil(torch.log2(amax / 6.0))
    best_deq, best_err = None, None
    for de in (-1.0, 0.0, 1.0):
        scale = torch.pow(2.0, e0 + de)
        v = wg / scale
        sign = torch.sign(v)
        a = v.abs().clamp(max=6.0)
        qa = grid[torch.bucketize(a, mids)]
        deq = sign * qa * scale
        err = ((deq - wg) ** 2).sum(dim=2, keepdim=True)
        if best_deq is None:
            best_deq, best_err = deq, err
        else:
            better = err < best_err
            best_deq = torch.where(better, deq, best_deq)
            best_err = torch.minimum(best_err, err)
    deq = best_deq.view(OUT, -1)[:, :IN]
    if omask is not None:
        deq = torch.where(omask, w32, deq)
    return deq
