"""ГЛУБИНА ЗАВИСИМОЙ ЦЕПОЧКИ ШАГА ДЕКОДА ПО ГРАФУ (24.09).

Граф выходов СКОМПИЛИРОВАННОГО шага (mx.export_to_dot, как probe_decode_graph_1909)
-> самый длинный путь по операциям. Два веса: raw (каждый узел = 1) и launch
(0 для заведомых видов/no-op: Reshape, Squeeze, ExpandDims, Broadcast, StopGradient,
AsStrided; Slice считается отдельно, т.к. может быть и видом, и копией).
Печать: всего узлов/запусков, длина пути, на слой, состав пути по типам, путь
одного среднего слоя (между LayerNorm ln1 слоя i и ln1 слоя i+1).
Плечи -- те же заглушки, что в bench_decode_rest_ab_2409 (текст скопирован оттуда).
Запуск: RWKVQ_COMP=<путь> RWKVQ_ARMS=full,L_gemv,... python probe_decode_depth_2409.py
ОГОВОРКА: длина пути в запусках -- верхняя граница числа ЗАВИСИМЫХ запусков; сколько
из них реально стоят ~17 мкс (19.09), граф не говорит -- это для лестницы.
"""
import os, sys, re, collections
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
import rwkv_quant.backends.metal.quant_linear_sym as sy
from rwkv_quant.formats.reader import load_raw

KV = "/Users/s/Develop/WKV-kvant/"
COMP = os.environ["RWKVQ_COMP"]
if not COMP.startswith("/"):
    COMP = KV + COMP
NOOP = {"Reshape", "Squeeze", "ExpandDims", "Broadcast", "StopGradient", "AsStrided"}


def pin(*arrs):
    s = None
    for a in arrs:
        if a is None:
            continue
        e = a.reshape(-1)[:1].astype(mx.float32)
        s = e if s is None else s + e
    return s * 0.0


def _lin_stub(self, x):
    return mx.zeros(x.shape[:-1] + (self.out_features,), mx.float32) + pin(x)


def _dense_stub(self, x):
    return mx.zeros(x.shape[:-1] + (int(self.w.shape[0]),), mx.float32) + pin(x)


def _aff_stub(self, x):
    return mx.zeros(x.shape[:-1] + (self.out_features,), mx.float32) + pin(x)


def _fused_stub(self, xstack):
    return mx.zeros((self.K, self.out_per), mx.float32) + pin(xstack)


ORIG = {"wkv": qm._wkv_stateful, "ln": qm._layer_norm, "tail": qm.wkv_tail,
        "lora": qm.QuantTMix._lora, "prewkv": qm.QuantTMix._prewkv,
        "gl": gw.GwQuantLinear.__call__, "gf": gw.GwQuantLinearFused.__call__,
        "sl": sy.SymQuantLinear.__call__, "sf": sy.SymQuantLinearFused.__call__,
        "fprewkv": qm.FUSE_PREWKV, "views": qm.DECODE_VIEWS,
        "dl": qm._DenseLinear.__call__, "al": qm.MlxAffineQuantLinear.__call__}


def apply(names):
    for n in names:
        if n == "wkv":
            qm._wkv_stateful = lambda r, w, k, v, a, b, state: (
                mx.zeros(r.shape, mx.float32) + pin(r, w, k, v, a, b), state + pin(r, w, k, v, a, b))
        elif n == "ln":
            qm._layer_norm = lambda x, weight, bias, eps=1e-5, fast=None: x
        elif n == "tail":
            qm.wkv_tail = lambda wkv, r, k, v, r_k, ln_w, ln_b, g, H, S, eps=64e-5: (
                wkv.reshape(-1, H * S) + pin(r, k, v, g))
        elif n == "lora":
            qm.QuantTMix._lora = lambda self, xw, xa, xv, xg, x, xx: (
                (mx.zeros_like(x) + pin(xw, xa, xv, xg, xx).astype(x.dtype),) * 4)
        elif n == "prewkv":
            qm.QuantTMix._prewkv = lambda self, y_w, y_a, y_v, k, v, v_first, B, T, dtype: (
                k + pin(y_w, y_a, y_v, v_first).astype(k.dtype), k, v, k, k, v)
        elif n == "views":
            qm.DECODE_VIEWS = True
        elif n == "fprewkv":
            qm.FUSE_PREWKV = True
        elif n == "gemv":
            gw.GwQuantLinear.__call__ = _lin_stub
            sy.SymQuantLinear.__call__ = _lin_stub
            gw.GwQuantLinearFused.__call__ = _fused_stub
            sy.SymQuantLinearFused.__call__ = _fused_stub
            qm._DenseLinear.__call__ = _dense_stub
            qm.MlxAffineQuantLinear.__call__ = _aff_stub


def restore():
    qm._wkv_stateful = ORIG["wkv"]; qm._layer_norm = ORIG["ln"]; qm.wkv_tail = ORIG["tail"]
    qm.QuantTMix._lora = ORIG["lora"]; qm.QuantTMix._prewkv = ORIG["prewkv"]
    gw.GwQuantLinear.__call__ = ORIG["gl"]; gw.GwQuantLinearFused.__call__ = ORIG["gf"]
    sy.SymQuantLinear.__call__ = ORIG["sl"]; sy.SymQuantLinearFused.__call__ = ORIG["sf"]
    qm.FUSE_PREWKV = ORIG["fprewkv"]
    qm.DECODE_VIEWS = ORIG["views"]
    qm._DenseLinear.__call__ = ORIG["dl"]; qm.MlxAffineQuantLinear.__call__ = ORIG["al"]


ARMS = {"full": [], "gemv0": ["gemv"], "all": ["gemv", "lora", "wkv", "ln", "tail", "prewkv"],
        "L_gemv": ["lora", "wkv", "ln", "tail", "prewkv"], "L_lora": ["wkv", "ln", "tail", "prewkv"],
        "L_wkvblk": ["lora", "ln"], "L_ln": ["lora", "wkv", "tail", "prewkv"],
        "lora0": ["lora"], "wkv0": ["wkv"], "prewkv_id": ["prewkv"], "tail_id": ["tail"], "ln_id": ["ln"]}
SEL = os.environ.get("RWKVQ_ARMS", "full").split(",")

m = qm.QuantRWKV7(load_raw(COMP))
NL = len(m.blocks)
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)


def graph(name):
    apply(ARMS[name])
    fn = mx.compile(lambda i, s: m.forward_stateful(i, s))
    st = m.init_state(1); tok = mx.array([[5]], dtype=mx.int32)
    for _ in range(2):
        lg, st = fn(tok, st); mx.eval(lg, st)
    lg, st = fn(tok, st)
    tok2 = mx.argmax(lg[:, -1], axis=-1)
    flat = [tok2] + [a for layer in st for a in (layer if isinstance(layer, (list, tuple)) else [layer])]
    out = "/tmp/x/depth_%s.dot" % name
    mx.export_to_dot(out, *flat)
    restore()
    txt = open(out).read()
    lab = dict(re.findall(r'\{ (\d+) \[label ="([^"]*)"', txt))
    succ_arr = collections.defaultdict(list)   # array -> ops
    op_out = collections.defaultdict(list)     # op -> arrays
    for src_, dst in re.findall(r'^\s*("?[\w]+"?) -> ("?[\w]+"?)', txt, re.M):
        s_, d_ = src_.strip('"'), dst.strip('"')
        if s_ in lab:
            op_out[s_].append(d_)
        else:
            succ_arr[s_].append(d_)
    succ = {o: [n for a in op_out[o] for n in succ_arr[a]] for o in lab}
    return lab, succ


def longest(lab, succ, w):
    indeg = collections.Counter(n for o in succ for n in succ[o])
    order, q = [], [o for o in lab if indeg[o] == 0]
    while q:
        o = q.pop(); order.append(o)
        for n in succ[o]:
            indeg[n] -= 1
            if indeg[n] == 0:
                q.append(n)
    assert len(order) == len(lab), "цикл в графе?"
    best = {o: w(lab[o]) for o in lab}; prev = {o: None for o in lab}
    for o in order:
        for n in succ[o]:
            if best[o] + w(lab[n]) > best[n]:
                best[n] = best[o] + w(lab[n]); prev[n] = o
    end = max(best, key=best.get)
    path = []
    while end is not None:
        path.append(end); end = prev[end]
    return path[::-1]


def short(k):
    return k.replace("Compiled", "C:")


print("=== %s, слоёв %d ===" % (os.path.basename(COMP), NL), flush=True)
for name in SEL:
    lab, succ = graph(name)
    kinds = collections.Counter(lab.values())
    nlaunch = sum(v for k, v in kinds.items() if k not in NOOP)
    p_raw = longest(lab, succ, lambda k: 1)
    p_l = longest(lab, succ, lambda k: 0 if k in NOOP else 1)
    L = sum(0 if lab[o] in NOOP else 1 for o in p_l)
    pk = collections.Counter(lab[o] for o in p_l if lab[o] not in NOOP)
    print("--- %s: узлов %d, запусков (не no-op) %d = %.1f/слой; путь raw %d узлов; путь по запускам %d = %.1f/слой" % (
        name, len(lab), nlaunch, nlaunch / NL, len(p_raw), L, L / NL), flush=True)
    print("    состав пути:", dict(pk.most_common(20)), flush=True)
    if name == "full":
        seq = [lab[o] for o in p_l if lab[o] not in NOOP]
        ln = [i for i, k in enumerate(seq) if k == "LayerNorm"]
        # ln0, затем по 2 на слой (ln1, ln2), затем ln_out
        if len(ln) >= 2 * NL + 1:
            i = NL // 2
            a_, b_ = ln[1 + 2 * i], ln[1 + 2 * (i + 1)]
            print("    путь слоя %d (%d запусков): %s" % (i, b_ - a_, " ".join(short(k) for k in seq[a_:b_])), flush=True)
        else:
            print("    LayerNorm на пути: %d (ожидалось >= %d) -- путь слоя не выделен" % (len(ln), 2 * NL + 1), flush=True)
    print("    типы узлов:", dict(kinds.most_common(14)), flush=True)
