"""05.10: разбор памяти Metal-бэкенда: (1) живые torch / numpy объекты после сборки модели (Malloc Large вне MLX);
(2) массивы MLX, достижимые из модели, по именам атрибутов -- до и после первого декода (что создаётся лениво).
    python mem_probe3_0510.py <файл.rwkvq>"""
import gc, os, re, sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal import generate as G
mx.set_cache_limit(64 * 2**20)
def walk(root):
    seen, out, stack = set(), {}, [("m", root)]
    while stack:
        path, o = stack.pop()
        if id(o) in seen: continue
        seen.add(id(o))
        if isinstance(o, mx.array):
            k = re.sub(r"\[\d+\]", "[N]", path); out.setdefault(k, [0, 0, str(o.dtype)]); out[k][0] += o.nbytes; out[k][1] += 1; continue
        if isinstance(o, (str, bytes, int, float, bool, type(None), np.ndarray, torch.Tensor)): continue
        if isinstance(o, dict):
            for k, v in o.items(): stack.append(("%s.%s" % (path, k), v))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o): stack.append(("%s[%d]" % (path, i), v))
        elif hasattr(o, "__dict__"):
            for k, v in vars(o).items(): stack.append(("%s.%s" % (path, k), v))
        if callable(o) and getattr(o, "__closure__", None):
            for i, c in enumerate(o.__closure__):
                try: stack.append(("%s<closure>" % path, c.cell_contents))
                except ValueError: pass
    return out
def big_host():
    gc.collect(); rows = []; seen = set()
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.numel() * o.element_size() > 8 * 2**20:
                p = o.untyped_storage().data_ptr()
                if p in seen: continue
                seen.add(p); rows.append((o.untyped_storage().nbytes(), "torch %s %s" % (tuple(o.shape), o.dtype)))
            elif isinstance(o, np.ndarray) and o.nbytes > 8 * 2**20:
                b = o
                while isinstance(b.base, np.ndarray): b = b.base
                if id(b) in seen: continue
                seen.add(id(b)); rows.append((b.nbytes, "numpy %s %s base=%s" % (o.shape, o.dtype, type(b.base).__name__)))
        except Exception: pass
    rows.sort(reverse=True)
    print("  живые torch / numpy > 8 МиБ: %d шт., всего %.0f МиБ" % (len(rows), sum(r[0] for r in rows) / 2**20))
    for n, d in rows[:12]: print("    %7.0f МиБ  %s" % (n / 2**20, d))
def show(tag, w, prev=None):
    tot = sum(v[0] for v in w.values())
    print("== %s: достижимо из модели %.0f МиБ в %d массивах; mx активная %.0f МиБ" % (tag, tot / 2**20, sum(v[1] for v in w.values()), mx.get_active_memory() / 2**20))
    items = sorted(w.items(), key=lambda kv: -kv[1][0])
    if prev is None:
        for k, v in items[:18]: print("    %8.1f МиБ x%-4d %-8s %s" % (v[0] / 2**20, v[1], v[2], k))
    else:
        d = {k: v[0] - prev.get(k, [0])[0] for k, v in w.items()}
        for k, x in sorted(d.items(), key=lambda kv: -abs(kv[1]))[:18]:
            if abs(x) > 2**20: print("    %+8.1f МиБ  x%-4d %-8s %s" % (x / 2**20, w[k][1], w[k][2], k))
raw = load_raw(sys.argv[1]); m = QuantRWKV7(raw); del raw; gc.collect()
big_host()
w0 = walk(m); show("после сборки", w0)
out = []
for y, _ in G.generate_step(m, [1, 2, 3, 4, 5, 6, 7, 8], 8, pipeline=True, out=out): pass
del out, y; gc.collect()
w1 = walk(m); show("после декода 8 токенов (изменения)", w1, w0)
big_host()
