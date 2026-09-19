"""ПОЛОСА ПО КОМПОНЕНТАМ ДЕКОДА (19.09): каждая GEMV-группа (rkv-фьюз, o,
cmix.key, cmix.value, голова) -- на настоящих весах модели, под mx.compile:
  indep -- все экземпляры группы (24 слоя) одним eval БЕЗ зависимостей:
           КПД самого кернеля на этой форме;
  chain -- те же 24 вызова ЗАВИСИМОЙ цепочкой: вход следующего = x0 + 0 *
           первый элемент выхода предыдущего (одна мелкая поэлементная
           операция на уровень -- её цена снята плечом addchain и вычтена);
  ideal -- зависимый читатель тех же байт по прямой 17 мкс + байты/107 ГБ/с
           (cache_chain_1909).
Голова -- один экземпляр, 24 вызова одного и того же тензора (94 МБ >> кэша).
Чередование групп и режимов по раундам, медиана.
"""
import os, sys, time, subprocess
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

COMP = "/Users/s/Develop/WKV-kvant/" + os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "9"))
m = qm.QuantRWKV7(load_raw(COMP))
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)
NL = len(m.blocks)
seen, keep = set(), []


def nbytes(l):
    s = 0
    for a in vars(l).values():
        if isinstance(a, mx.array):
            keep.append(a); s += a.nbytes
    return s


G = {}
t0 = m.blocks[1].tmix
if t0._rkv_fused is not None:
    G["rkv"] = ([b.tmix._rkv_fused for b in m.blocks], 3, t0._rkv_fused.in_features)
G["o"] = ([b.tmix.o_proj for b in m.blocks], 0, m.blocks[0].tmix.o_proj.in_features)
G["key"] = ([b.cmix.key for b in m.blocks], 0, m.blocks[0].cmix.key.in_features)
G["value"] = ([b.cmix.value for b in m.blocks], 0, m.blocks[0].cmix.value.in_features)
G["head"] = ([m.head] * NL, 0, m.head.in_features)

rng = np.random.default_rng(0)
X = {}
for g, (lins, K, IN) in G.items():
    shp = (K, IN) if K else (1, IN)
    X[g] = [mx.array(rng.standard_normal(shp).astype(np.float32) * 0.1) for _ in range(NL)]
mx.eval([a for v in X.values() for a in v])


def indep_fn(g):
    lins = G[g][0]
    def f(*xs):
        outs = [l(x) for l, x in zip(lins, xs)]
        return sum(o.reshape(-1)[:1] for o in outs)
    return mx.compile(f)


def chain_fn(g):
    lins = G[g][0]
    def f(*xs):
        dep = None
        for l, x in zip(lins, xs):
            xin = x if dep is None else x + dep * 0.0
            o = l(xin)
            dep = o.reshape(-1)[:1]
        return dep
    return mx.compile(f)


def addchain_fn(g):
    def f(*xs):
        dep = None
        for x in xs:
            xin = x if dep is None else x + dep * 0.0
            dep = xin.reshape(-1)[:1] * 1.0001
        return dep
    return mx.compile(f)


FN = {}
for g in G:
    FN[(g, "indep")] = indep_fn(g); FN[(g, "chain")] = chain_fn(g); FN[(g, "addchain")] = addchain_fn(g)


def t(key, reps=7):
    f = FN[key]; xs = X[key[0]]
    mx.eval(f(*xs)); mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize(); a = time.perf_counter(); mx.eval(f(*xs)); mx.synchronize()
        ts.append(time.perf_counter() - a)
    return float(np.median(ts)) * 1e6


def swap():
    return float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))


sw0 = swap()
keys = list(FN)
V = {k: [] for k in keys}
for rd in range(ROUNDS):
    for k in (keys if rd % 2 == 0 else keys[::-1]):
        V[k].append(t(k))

print("=== %s, %d слоёв ===" % (os.path.basename(COMP), NL))
print("группа | МБ/вызов | форма | indep мкс ГБ/с %% | chain(-add) мкс ГБ/с %% | ideal мкс %% | chain/ideal | на токен мс (chain)")
tot_chain = 0.0
for g, (lins, K, IN) in G.items():
    B = nbytes(lins[0]) / 1e6
    OUT = lins[0].out_features
    ind = np.median(V[(g, "indep")]) / NL
    add = np.median(V[(g, "addchain")]) / NL
    ch = np.median(V[(g, "chain")]) / NL - add
    idl = 17 + B / 107 * 1e3
    sp = 100 * (max(V[(g, "chain")]) - min(V[(g, "chain")])) / np.median(V[(g, "chain")])
    ntok = NL if g != "head" else 1
    tot_chain += ch * ntok / 1e3
    print("%-6s | %6.2f | %s%dx%d | %6.1f %5.1f %3.0f%% | %6.1f %5.1f %3.0f%% (разброс %.1f%%) | %6.1f %3.0f%% | %.2f | %.3f" % (
        g, B, ("%dx " % K) if K else "", IN, OUT if not K else lins[0].out_per,
        ind, B / ind * 1e3, 100 * B / ind * 1e3 / 104,
        ch, B / ch * 1e3, 100 * B / ch * 1e3 / 104, sp,
        idl, 100 * B / idl * 1e3 / 104, ch / idl, ch * ntok / 1e3))
print("сумма по цепочкам (на токен): %.3f мс; цена добавочной операции на уровень: см. addchain" % tot_chain)
for g in G:
    print("  addchain %-6s %.1f мкс/уровень" % (g, np.median(V[(g, "addchain")]) / NL))
print("своп %+.1f МБ" % (swap() - sw0))
