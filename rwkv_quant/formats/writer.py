"""
Настоящее (не fake) квантование + сохранение в .rwkvq.

Осознанно отделено от calibration.fake_quant: там задача -- измерить
ppl-эффект (дёшево, дёргается тысячи раз при ablation), здесь -- один раз
произвести реальные упакованные коды для сохранения на диск.
"""
import torch

from ..calibration.group_config import QuantConfig
from ..calibration.outlier_scan import GROUP_KEY_PATTERNS
from .schema import (QuantizedTensor, QuantizedCheckpoint, pack_int4,
                     pack6, pack_nib_block, pack_bitplane)


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


_ACT_STATS_CACHE = {}


def _load_act_stats(path):
    if path not in _ACT_STATS_CACHE:
        _ACT_STATS_CACHE[path] = torch.load(path)
    return _ACT_STATS_CACHE[path]


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


def _groupwise_fake_dequant(w: torch.Tensor, bits: int, gs: int,
                            sb: int = 0, sb_bits: int = 6, ex2=None,
                            return_parts: bool = False):
    """ПРОТОТИП group-wise scale (ядро K-квантов): асимметричный RTN на блок
    из gs колонок (scale + min на блок), сразу деквантованный обратно.
    Хранится как dense bf16 -> реальный пайплайн меряет ppl именно этой
    схемы без написания формата/кернеля. НЕ для продакшена: размер dense.
    Оверхед будущего формата: 2xfp16 на gs=32 элементов = +1 бит/элемент
    (int4 -> eff 5.0 бит); с fp8-scale или Q4_K-style суперблоками меньше."""
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
        # AW: вес колонки = E[x^2] её входного канала (см. №4f). Влияет
        # только на критерий поиска и LS, формат/раскладка не меняются.
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
            q = torch.clamp(torch.round((wg - mn) / sc), 0, qmax)
            # LS: min_{s,m} sum (w - q*s - m)^2 на блок
            if evg is None:
                qm_ = q.mean(dim=2, keepdim=True); wm_ = wg.mean(dim=2, keepdim=True)
                cov = ((q - qm_) * (wg - wm_)).sum(dim=2, keepdim=True)
                var = ((q - qm_) ** 2).sum(dim=2, keepdim=True).clamp_min(1e-12)
            else:  # взвешенная LS: те же формулы со средними по весам evg
                wsum = evg.sum(dim=2, keepdim=True)
                qm_ = (evg * q).sum(dim=2, keepdim=True) / wsum
                wm_ = (evg * wg).sum(dim=2, keepdim=True) / wsum
                cov = (evg * (q - qm_) * (wg - wm_)).sum(dim=2, keepdim=True)
                var = (evg * (q - qm_) ** 2).sum(dim=2, keepdim=True).clamp_min(1e-12)
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
        # scale частично компенсируется выбором кодов. Бюджет при gs=32,
        # sb=8, sb_bits=6: 4 + (2*6)/32 + (2*16)/256 = 4.5 бит/элемент.
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
    q = torch.clamp(torch.round((wg - mn) / scale), 0, qmax)
    deq = (q * scale + mn).view(OUT, -1)[:, :IN]
    if not return_parts:
        return deq
    parts = {"q": q.view(OUT, -1)[:, :IN].to(torch.uint8), "deq": deq,
             "scale": scale, "mn": mn}
    if sb:
        parts.update(qs=qs.view(OUT, -1, 1)[:, :nb].squeeze(-1).to(torch.uint8),
                     qm=qm.view(OUT, -1, 1)[:, :nb].squeeze(-1).to(torch.int8),
                     d=d.view(OUT, -1).half(), dm=dm.view(OUT, -1).half())
    return parts




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


_E2M1_GRID = None
def _mxfp4_fake_dequant(w: torch.Tensor, gs: int, outlier_frac: float = 0.0) -> torch.Tensor:
    """ПРОТОТИП MXFP4 (OCP MX): блок gs колонок, shared E8M0 scale (степень
    двойки), элементы FP4 E2M1 (+-{0,.5,1,1.5,2,3,4,6}). Экспонента блока
    выбирается перебором e0-1/e0/e0+1 по MSE блока (e0 = минимальная без
    клиппинга) -- честный best-effort, симметрично scale-гриду асимметричного
    RTN (_groupwise_fake_dequant). Выход dense bf16 (fake-dequant), формат и
    кернель не пишутся до валидации идеи. Эфф. битность: 4 + 8/gs = 4.25
    бит/элемент при gs=32 (против 5.0 у асимметричного gw32 c 2xfp16)."""
    global _E2M1_GRID
    if _E2M1_GRID is None:
        _E2M1_GRID = (torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.]),
                      torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.]))
    grid, mids = _E2M1_GRID
    w32 = w.float()
    OUT, IN = w32.shape
    omask = None
    if outlier_frac > 0.0:
        # SpQR-семантика как в _real_quantize_sparse_outlier: per-row top-k
        # по |w|, выбросы уходят в sparse (здесь: возвращаются verbatim),
        # блоки квантуют остаток -- amax/экспонента без хвостов.
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
                ex2 = (_load_act_stats(sp).get(key)
                       if (sp and mode == "asym_sb6_aw") else None)
                return _make_qt_gw_sb6(key, group, bits, w, gs, ex2,
                                       search=(mode != "asym_sb6"))
            if mode == "asym" and 5 <= bits <= 8:
                return _make_qt_gw_asym(key, group, bits, w, gs)
            raise NotImplementedError(f"real_gw: mode={mode} bits={bits} ({key})")
        if mode == "mxfp4":
            deq = _mxfp4_fake_dequant(w, gs, cfg.outlier_fracs.get(group, 0.0))
        elif mode == "asym_sb6":
            deq = _groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=6)
        elif mode == "asym_sb6_search":
            deq = _groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=-6)
        elif mode == "asym_sb6_aw":
            ex2 = _load_act_stats(sp).get(key) if sp else None
            deq = _groupwise_fake_dequant(w, bits, gs, sb=8, sb_bits=-6, ex2=ex2)
        else:
            deq = _groupwise_fake_dequant(w, bits, gs)
        return QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape),
                                dense=deq.to(torch.bfloat16))
    if sp and bits < 16:
        stats = _load_act_stats(sp)
        if key in stats:
            frac = cfg.outlier_fracs.get(group, 0.0)
            codes, scale, oi, ov = _weighted_quantize(w, bits, stats[key], frac)
            return _make_qt(key, group, bits, w.shape, codes, scale, oi, ov)
        # нет статистики (emb: вход -- индексы токенов; LoRA не писали) --
        # проваливаемся в обычный путь ниже
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
