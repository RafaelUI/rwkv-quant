"""ГЕЙТ однопроходного кернеля декванта: БИТ-В-БИТ против эталонной
цепочки `_dequant_w_ref` на ВСЕХ тензорах реального файла (закон 17: формы из
настоящей модели, закон 31: обе разновидности xbits покрыты).

Порог бит-в-бит здесь не выбран, а вынужден: кернель обязан отдавать тот
же тензор, что плотный путь, иначе меняется референс всего проекта.
Печатается и ULP-расстояние -- если бит-в-бит не выйдет, нужно знать, на
сколько именно (1 ULP -- это решение владельца, а не повод править порог).

РАЗРЕШАЮЩАЯ СПОСОБНОСТЬ ГЕЙТА МЕРИТСЯ МУТАЦИЕЙ (закон 37):
    MUTATE=1 python tests/test_gw_dequant_kernel_parity.py
портит сдвиг битплоскости (4 -> 5) и гейт ОБЯЗАН покраснеть. Если он при
этом зелёный -- он ничего не проверяет.

    python tests/test_gw_dequant_kernel_parity.py [model.rwkvq]
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mlx.core as mx

from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
import rwkv_quant.backends.metal.quant_linear_gw as gw
import rwkv_quant.backends.metal.gw_dequant_kernel as gdk

# ФАКТ ВКЛЮЧЕНИЯ (17.09): рабочий _dequant_w обязан уходить в кернель на
# каждом допустимом тензоре и отдавать ровно его выход. Счётчик ставится
# на атрибут модуля -- метод берёт его ленивым импортом при вызове.
dequant_w = gdk.dequant_w
ROUTED = [0]


def _counted(self, mutate=False):
    ROUTED[0] += 1
    return dequant_w(self, mutate)


gdk.dequant_w = _counted

REF = gw.GwQuantLinear._dequant_w_ref
MUT = bool(int(os.environ.get("MUTATE", "0")))
PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/Users/s/Develop/WKV-kvant/compression_v2_cand.rwkvq")


def collect(root):
    seen, out, stack = set(), [], [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, gw.GwQuantLinear):
            out.append(o)
            continue
        if isinstance(o, (list, tuple)):
            stack.extend(o)
            continue
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            stack.extend(d.values())
    return out


print("файл %s, мутация %s" % (os.path.basename(PATH), "ВКЛ" if MUT else "выкл"), flush=True)
m = QuantRWKV7(load_raw(PATH))
lins = collect(m)
print("тензоров %d" % len(lins), flush=True)
bad, worst_ulp, worst_abs, cnt = [], 0, 0.0, {}
routed, wrong = 0, []
for l in lins:
    a, b = REF(l), dequant_w(l, mutate=MUT)
    n0 = ROUTED[0]
    c = l._dequant_w()
    mx.eval(a, b, c)
    went = ROUTED[0] > n0
    elig = bool(l._k3) and l.xbits <= 1 and l.in_features % 256 == 0
    if went != elig:
        wrong.append((l.out_features, l.in_features, l.xbits, went))
    if went:
        routed += 1
        if not MUT:
            assert np.array_equal(np.array(mx.view(b, mx.uint16), copy=False),
                                  np.array(mx.view(c, mx.uint16), copy=False)), (
                "рабочий _dequant_w отдал не выход кернеля", l.out_features)
    del c
    assert a.shape == b.shape and a.dtype == b.dtype, ("форма или тип", l.out_features)
    ua = np.array(mx.view(a, mx.uint16), copy=False).astype(np.int64)
    ub = np.array(mx.view(b, mx.uint16), copy=False).astype(np.int64)
    ne = int(np.count_nonzero(ua != ub))
    k = l.xbits
    cnt[k] = cnt.get(k, 0) + 1
    if ne:
        u = int(np.abs(ua - ub).max())
        d = float(np.abs(np.array(a, copy=False).astype(np.float32)
                         - np.array(b, copy=False).astype(np.float32)).max())
        worst_ulp = max(worst_ulp, u)
        worst_abs = max(worst_abs, d)
        bad.append((l.out_features, l.in_features, k, ne, u))
    del a, b, ua, ub
print("  рабочий _dequant_w ушёл в кернель на %d из %d; маршрут неверен на %d" % (
    routed, len(lins), len(wrong)), flush=True)
assert not wrong, ("маршрут _dequant_w не совпал с условием включения", wrong[:4])
for k in sorted(cnt):
    print("  xbits=%d: %d тензоров" % (k, cnt[k]), flush=True)
if bad:
    print("  РАСХОЖДЕНИЯ на %d тензорах из %d; худшее %d ULP, макс abs %.3e" % (
        len(bad), len(lins), worst_ulp, worst_abs), flush=True)
    for t in bad[:4]:
        print("    OUT=%d IN=%d xbits=%d: %d элементов, до %d ULP" % t, flush=True)
else:
    print("  расхождений НЕТ ни на одном элементе", flush=True)
if MUT:
    assert bad, "МУТАЦИЯ НЕ ПОЙМАНА: гейт слеп, доверять ему нельзя"
    print("ГЕЙТ КРАСНЫЙ НА МУТАЦИИ -- разрешающая способность подтверждена", flush=True)
else:
    assert not bad, "кернель НЕ бит-в-бит против штатного пути"
    print("ГЕЙТ ЗЕЛЁНЫЙ: бит-в-бит на всех тензорах", flush=True)
