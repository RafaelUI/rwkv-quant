# -*- coding: utf-8 -*-
"""ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ К УТВЕРЖДЕНИЮ "AW к emb не применялся".

Утверждение выведено из множества ключей (в act_stats нет emb.weight), а
не измерено. Здесь оно проверяется прямо: РЕАЛЬНЫЙ emb.weight пакуется
дважды -- режимом `sym_aw` и режимом `sym` -- и сравниваются БАЙТЫ.

Что означает каждый исход:
  байты равны   -> AW действительно не применялся: два разных режима
                   дают один результат, то есть `_aw` на emb -- надпись;
  байты разные  -> вывод неверен, статистика где-то находится.

Контроль осмысленности: тот же прогон на head.weight, где статистика ЕСТЬ.
Там байты ОБЯЗАНЫ разойтись -- иначе инструмент не отличает режимы вовсе
и первый результат ничего не значит (закон 8: ложный бит-экзакт).
"""
import os, sys, copy, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from rwkv_quant.presets import REDUCTION
from rwkv_quant.formats.writer import quantize_tensor, TENSOR_FIELDS

CKPT  = "/Users/s/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"
STATS = "/tmp/act_stats_1p5b_ml.pt"

def digest(qt):
    h = hashlib.md5()
    for f in TENSOR_FIELDS:
        v = getattr(qt, f, None)
        h.update(b"\x00" if v is None else v.contiguous().numpy().tobytes())
    return h.hexdigest()

def cfg_with(group, mode):
    c = copy.deepcopy(REDUCTION)
    c.act_stats_path = STATS
    c.group_scale_mode[group] = mode
    return c

sd = torch.load(CKPT, map_location="cpu", mmap=True)
for key, group in (("emb.weight", "emb"), ("head.weight", "head")):
    w = sd[key]
    a = digest(quantize_tensor(key, w, cfg_with(group, "sym_aw"), real_gw=True))
    b = digest(quantize_tensor(key, w, cfg_with(group, "sym"),    real_gw=True))
    print("%-12s %-8s sym_aw=%s  sym=%s  ->  %s"
          % (key, group, a[:12], b[:12],
             "БАЙТ-В-БАЙТ (AW не сработал)" if a == b else "РАЗОШЛИСЬ (AW сработал)"),
          flush=True)
