# -*- coding: utf-8 -*-
"""ЛИШНЕЕ ОКРУГЛЕНИЕ В ПЛОТНОМ ПУТИ: СТОИТ ЛИ ОНО ЧЕГО-НИБУДЬ (приоритет 5).

`_dense` считает деквант в fp32, округляет в bf16 (reader) и лишь потом
кладёт в fp16-контейнер. Записано: "таблица несёт точность bf16 в
fp16-контейнере, два бита мантиссы бесплатно".

ПРОВЕРЯЮ ГИПОТЕЗУ, ГОВОРЯЩУЮ ОБРАТНОЕ. Чекпоинт сам bf16, то есть ИСТИНА
ЛЕЖИТ НА СЕТКЕ bf16. Если ошибка квантования меньше половины ulp bf16,
округление реконструкции в bf16 не портит её, а ВОЗВРАЩАЕТ на исходное
значение -- то есть работает шумодавом, и снятие bf16 сделает ХУЖЕ. Что
из двух верно, решает соотношение шага квантования и шага bf16.

ppl тут не инструмент: полоса различимости +-0.05 п.п. против ожидаемого
эффекта в сотые. Меряется САМА ОШИБКА против оригинального веса, в fp64.

ГРАБЛИ, СТОИВШИЕ БЫ ВСЕГО ЗАМЕРА: `_dequantize_one` и `_dequantize_gw_sym`
САМИ заканчиваются `.to(bfloat16)`, поэтому взять их вывод за "пол fp32"
нельзя -- все три плеча оказались бы одним числом, и вывод "эффекта нет"
был бы артефактом инструмента. Ниже fp32-деквант выписан отдельно, и его
тождественность проверяется ПОЛОЖИТЕЛЬНЫМ КОНТРОЛЕМ: fp32-версия,
округлённая в bf16, обязана совпасть с библиотечной БИТ-В-БИТ.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from rwkv_quant.formats.reader import (load_raw, _dequantize_gw_sym,
                                       unpack_nib_block, unpack_bitplane)

PAIRS = [("1.5B", "/tmp/reduction_new.rwkvq",
          "/Users/s/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"),
         ("2.9B", "/tmp/reduction_2p9b_new.rwkvq",
          "/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth")]


def deq_fp32(qt):
    """Копия _dequantize_gw_sym БЕЗ финального каста -- одна изменённая
    строка, всё остальное посимвольно то же (закон 15)."""
    OUT, IN = qt.shape
    gs, NB = qt.gw_gs, IN // qt.gw_gs
    if qt.codes is not None:
        q = qt.codes.to(torch.float32)
    else:
        q = unpack_nib_block(qt.codes_packed, gs).to(torch.int16)
        if qt.gw_qh is not None:
            q = q + unpack_bitplane(qt.gw_qh, IN).to(torch.int16) * 16
        if qt.gw_qh2 is not None:
            q = q + unpack_bitplane(qt.gw_qh2, IN).to(torch.int16) * 32
        q = (q - 32).to(torch.float32)
    d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)
    scale = (qt.gw_qs.float() * d).half().float()
    if NB * gs == IN:
        q = q.view(OUT, NB, gs)
        q.mul_(scale[..., None])
        return q.view(OUT, IN)
    return q * scale.repeat_interleave(gs, dim=1)


def rms(x):
    return float(torch.sqrt((x.double() ** 2).mean()))


for tag, qpath, ckpt in PAIRS:
    ck = load_raw(qpath)
    sd = torch.load(ckpt, map_location="cpu", mmap=True)
    keys = [k for k in ("emb.weight", "head.weight", "blocks.0.att.key.weight")
            if k in ck.tensors and getattr(ck.tensors[k], "gw_mode", "") == "sym"]
    print("\n=== %s ===" % tag, flush=True)
    print("%-26s %11s %11s %11s %8s %9s"
          % ("тензор", "пол fp32", "A bf16→fp16", "B fp16", "A/пол", "bf16 ближе"))
    for key in keys:
        qt = ck.tensors[key]
        w0 = sd[key].float()
        raw = deq_fp32(qt)
        # ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ: без него "пол" мог бы считаться другим кодом
        lib = _dequantize_gw_sym(qt)
        assert torch.equal(raw.to(torch.bfloat16), lib), "fp32-деквант разошёлся с библиотечным: %s" % key
        del lib
        a = raw.to(torch.bfloat16).to(torch.float16).float()
        b = raw.to(torch.float16).float()
        e_raw, e_a, e_b = (raw - w0).abs(), (a - w0).abs(), (b - w0).abs()
        closer = float((e_a < e_raw).double().mean())
        print("%-26s %11.4e %11.4e %11.4e %7.1f%% %8.1f%%"
              % (key, rms(raw - w0), rms(a - w0), rms(b - w0),
                 100 * rms(a - w0) / rms(raw - w0), 100 * closer), flush=True)
        del w0, raw, a, b, e_raw, e_a, e_b
    del sd, ck
