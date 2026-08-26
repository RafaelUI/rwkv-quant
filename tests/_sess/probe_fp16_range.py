# -*- coding: utf-8 -*-
"""ПОЛ ПУТИ 0.000575: НЕ ЭКСПОНЕНТА ЛИ fp16 (гипотеза, проверяется прямо).

`_dense` кладёт 2D-тензоры в fp16. Мантисса там ДЛИННЕЕ, чем у bf16
(10 бит против 8), поэтому перевод выглядит безобидным. Но экспонента
КОРОЧЕ: минимальное нормальное fp16 -- 6.10e-05, ниже идут субнормали с
убывающей точностью, а с 5.96e-08 -- ноль. bf16 держит те же 1e-38, что
и fp32. Если в чекпоинте есть заметная доля весов ниже 6.10e-05, то
fp16-контейнер их портит, и это объясняет пол пути лучше, чем мантисса.

Меряется доля весов в трёх зонах и ошибка роундтрипа bf16 -> fp16 -> fp32.
ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ: та же ошибка для bf16 -> bf16 обязана быть НОЛЬ,
иначе инструмент меряет не то.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch

CKPT = os.environ.get("RWKVQ_CKPT",
                      "/Users/s/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
SUBN, ZERO = 6.103515625e-05, 5.960464477539063e-08

sd = torch.load(CKPT, map_location="cpu", mmap=True)
keys = [k for k in ("emb.weight", "head.weight", "blocks.0.att.receptance.weight",
                    "blocks.0.att.w1", "blocks.0.att.w2", "blocks.0.att.a1",
                    "blocks.0.att.g1", "blocks.0.ffn.key.weight") if k in sd]
print("%-32s %9s %9s %9s %11s %11s"
      % ("тензор", "<6.1e-5", "<5.96e-8", "==0 было", "RMS fp16", "RMS bf16"))
tot_sub = tot_n = 0
for k in keys:
    w = sd[k]
    if w.ndim != 2:
        continue
    f = w.float()
    a = f.abs()
    n = a.numel()
    sub = int((a < SUBN).sum()) - int((a == 0).sum())
    zer = int(((a < ZERO) & (a > 0)).sum())
    was0 = int((a == 0).sum())
    e16 = float(torch.sqrt(((f.to(torch.float16).float() - f).double() ** 2).mean()))
    e_bf = float(torch.sqrt(((f.to(torch.bfloat16).float() - f).double() ** 2).mean()))
    tot_sub += sub; tot_n += n
    print("%-32s %8.3f%% %8.3f%% %8.3f%% %11.4e %11.4e"
          % (k, 100*sub/n, 100*zer/n, 100*was0/n, e16, e_bf), flush=True)
    del w, f, a
print("\nсубнормалей суммарно по показанным тензорам: %.3f%% (%d из %d)"
      % (100*tot_sub/tot_n, tot_sub, tot_n))
print("ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ: RMS bf16 обязана быть 0 -- чекпоинт УЖЕ bf16;")
print("если она не ноль, инструмент меряет не роундтрип, а что-то другое.")
