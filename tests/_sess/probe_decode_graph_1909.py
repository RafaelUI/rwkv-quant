"""Состав шага декода по примитивам: граф выходов СКОМПИЛИРОВАННОГО шага
(mx.export_to_dot) -> счёт узлов по типу. Узел примитива ~ один диспатч
(кроме no-op: reshape/slice-вьюхи могут не запускать кернель). Грубый, но
прямой счёт вместо вычитания."""
import sys, re, collections
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw
m = qm.QuantRWKV7(load_raw("/Users/s/Develop/WKV-kvant/" + sys.argv[1]))
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)
fn = mx.compile(lambda i, s: m.forward_stateful(i, s))
st = m.init_state(1); tok = mx.array([[5]], dtype=mx.int32)
for _ in range(2):
    lg, st = fn(tok, st); mx.eval(lg, st)
lg, st = fn(tok, st)
tok2 = mx.argmax(lg[:, -1], axis=-1)
flat = [tok2] + [a for layer in st for a in (layer if isinstance(layer, (list, tuple)) else [layer])]
out = "/tmp/x/decode_graph.dot"
mx.export_to_dot(out, *flat)
txt = open(out).read()
labels = re.findall(r'label\s*=\s*"([^"]*)"', txt)
prims = [l.split("\\n")[0].split()[0] for l in labels]
c = collections.Counter(prims)
print(sys.argv[1], "узлов с меткой:", len(labels))
for k, v in c.most_common(40):
    print("  %6d  %s" % (v, k))
