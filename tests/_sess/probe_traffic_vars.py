"""Что линейный слой ХРАНИТ (а не отдаёт свойством) и сколько это байт.

Счётчик trace_decode_steady.decode_traffic_mb перечисляет поля списком имён
и дедуплицирует по id() неудерживаемого объекта. 09.09 выяснено: qs, d, codes
-- свойства, отдающие свежий массив, поэтому дедуп ложный, а ответ плавает.
Здесь трафик считается по vars(): только реально хранимые буферы, каждый один
раз. Сверка -- с размером файла на диске за вычетом emb.
"""
import os
import sys

sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")

import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as _qm
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]


def dump(name, obj):
    print("--- " + name + " ---", flush=True)
    tot = 0
    for k, v in sorted(vars(obj).items()):
        if isinstance(v, mx.array):
            tot += v.nbytes
            print("  %-14s %-16s %8.3f МБ" % (k, str(v.shape), v.nbytes / 1e6), flush=True)
    print("  ИТОГО хранимого: %.3f МБ" % (tot / 1e6), flush=True)
    return tot


def stored(obj, seen, out):
    for v in vars(obj).values():
        if isinstance(v, mx.array) and id(v) not in seen:
            seen.add(id(v))
            out.append(v)


def main():
    m = QuantRWKV7(load_raw(PATH))
    print("файл: %s, на диске %.2f МБ" % (os.path.basename(PATH), os.path.getsize(PATH) / 1e6), flush=True)
    dump("r_proj слоя 0", m.blocks[0].tmix.r_proj)
    dump("cmix.key слоя 0", m.blocks[0].cmix.key)
    dump("head", m.head)
    seen, out = set(), []
    stored(m.head, seen, out)
    for b in m.blocks:
        tm, cm = b.tmix, b.cmix
        for l in (tm.r_proj, tm.k_proj, tm.v_proj, tm.o_proj, cm.key, cm.value):
            stored(l, seen, out)
        stored(tm, seen, out)
        stored(cm, seen, out)
        stored(b, seen, out)
    tot = sum(a.nbytes for a in out)
    print("", flush=True)
    print("ТРАФИК по vars(): %.1f МБ, тензоров %d" % (tot / 1e6, len(out)), flush=True)
    emb = m.emb_weight
    print("emb как массив: %s, %.1f МБ (в трафик НЕ входит -- gather строки)"
          % (str(emb.shape), emb.nbytes / 1e6), flush=True)
    print("сверка: файл %.1f - emb на диске ~= трафик"
          % (os.path.getsize(PATH) / 1e6,), flush=True)


if __name__ == "__main__":
    main()
