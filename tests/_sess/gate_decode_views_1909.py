"""Гейт DECODE_VIEWS (19.09): декод с флагом обязан давать ЛОГИТЫ БИТ-В-БИТ
с путём без флага на каждом шаге жадной траектории (64 шага от пустого
состояния, оба под mx.compile). Разрешение меряется мутацией: в одном слое
переставлены строки w и v стека коэффициентов -- гейт обязан покраснеть.
Плюс счёт примитивов шага: Gather и Concatenate должны уйти."""
import os, sys, re, collections
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw
m = qm.QuantRWKV7(load_raw("/Users/s/Develop/WKV-kvant/" + sys.argv[1]))
N = 64


def compiled(flag):
    qm.DECODE_VIEWS = flag
    fn = mx.compile(lambda i, s: m.forward_stateful(i, s))
    lg, _ = fn(mx.array([[3]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)
    qm.DECODE_VIEWS = False
    return fn


def run(fn):
    st, tok, out = m.init_state(1), mx.array([7], dtype=mx.int32), []
    for _ in range(N):
        lg, st = fn(tok[None], st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(lg, st)
        out.append(lg)
    return out


def prims(fn):
    lg, st = fn(mx.array([[5]], dtype=mx.int32), m.init_state(1))
    mx.export_to_dot("/tmp/x/g.dot", lg)
    return collections.Counter(re.findall(r'label ="([^"]*)"', open("/tmp/x/g.dot").read()))


ref, new = compiled(False), compiled(True)
a, b = run(ref), run(new)
bad = [i for i in range(N) if not mx.array_equal(a[i], b[i]).item()]
pa, pb = prims(ref), prims(new)
print(sys.argv[1], "шагов %d, несовпавших %d" % (N, len(bad)), "-> ЗЕЛЁНЫЙ" if not bad else "-> КРАСНЫЙ", flush=True)
for k in ("Gather", "Concatenate", "Slice", "CustomKernel", "QuantizedMatmul"):
    print("  %-16s %5d -> %5d" % (k, pa.get(k, 0), pb.get(k, 0)))
tot = lambda c: sum(v for k, v in c.items() if k not in ("Reshape", "Squeeze", "ExpandDims", "Slice"))
print("  запусков (без видов) %d -> %d" % (tot(pa), tot(pb)))
# мутация: строки w и v в стеке одного слоя
t = m.blocks[1].tmix
xc = t._xcoef_v
t._xcoef_v = mx.stack([xc[0], xc[1], xc[3], xc[2], xc[4], xc[5]])
mut = compiled(True)
c = run(mut)
badm = sum(1 for i in range(N) if not mx.array_equal(a[i], c[i]).item())
print("  мутация (w<->v в слое 1): несовпавших %d -> %s" % (badm, "ПОЙМАНА" if badm else "НЕ ПОЙМАНА -- ГЕЙТ СЛЕП"))
