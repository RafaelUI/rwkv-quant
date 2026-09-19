"""ПОТОЛОК «ВСЁ, КРОМЕ GEMV, БЕСПЛАТНО» (19.09): скомпилированная цепочка
ровно тех GEMV, что в настоящем шаге декода (фьюз rkv, o, key, value на
слой, голова, argmax), в ритме eval каждый токен, чередуется с настоящим
шагом. Разница full - gemv_chain = всё, что можно выиграть, НЕ трогая сами
GEMV-кернели; ГБ/с цепочки = полоса, выше которой без перестройки GEMV не
подняться.

Отличия от шага: вход фьюза собран concatenate из трёх копий (1 лишняя
мелкая операция на слой, в пользу шага), между key и value нет relu^2.
Чтобы значения не улетали в inf, между GEMV стоит умножение на константу
(сливается компилятором с соседним reshape, отдельного запуска не даёт --
проверяется счётом примитивов в конце).
"""
import os, sys, time, subprocess, re, collections
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

KV = "/Users/s/Develop/WKV-kvant/"
COMP = KV + os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")
K = int(os.environ.get("RWKVQ_K", "32")); ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "11"))


def swap_used():
    return float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))


sw0 = swap_used()
m = qm.QuantRWKV7(load_raw(COMP))
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)
D = int(m.emb_weight.shape[1])
EMB = m.emb_weight


def chain(tok):
    y = EMB[tok].reshape(1, D).astype(mx.float32)
    for b in m.blocks:
        t, c = b.tmix, b.cmix
        if t._rkv_fused is not None:
            rkv = t._rkv_fused(mx.concatenate([y, y, y], axis=0))
            r = rkv[0:1]
        else:
            r = t.r_proj(y)
        y = t.o_proj(r * 0.03)
        h = c.key(y * 0.03)
        y = c.value(h * 0.001)
    return mx.argmax(m.head(y), axis=-1)


chain_c = mx.compile(chain)
full_c = mx.compile(lambda i, s: m.forward_stateful(i, s))

st_f = [m.init_state(1), mx.array([7], dtype=mx.int32)]
tok_c = [mx.array([7], dtype=mx.int32)]


def run_full():
    st, tok = st_f
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(K):
        lg, st = full_c(tok[None], st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok, st)
    mx.synchronize(); dt = (time.perf_counter() - t0) * 1e3 / K
    st_f[0], st_f[1] = st, tok
    return dt


def run_chain():
    tok = tok_c[0]
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(K):
        tok = chain_c(tok)
        mx.eval(tok)
    mx.synchronize(); dt = (time.perf_counter() - t0) * 1e3 / K
    tok_c[0] = tok
    return dt


for _ in range(2):
    run_full(); run_chain()
V = {"full": [], "gemv_chain": []}
for rd in range(ROUNDS):
    order = [("full", run_full), ("gemv_chain", run_chain)]
    for name, f in (order if rd % 2 == 0 else order[::-1]):
        V[name].append(f())

# счёт примитивов цепочки: сколько запусков на слой она реально делает
mx.export_to_dot("/tmp/x/chain.dot", chain_c(tok_c[0]))
labs = [l for l in re.findall(r'label ="([^"]*)"', open("/tmp/x/chain.dot").read())]
cnt = collections.Counter(labs)

# байты GEMV-фазы
seen, keep = set(), []
def nb(objs):
    s = 0
    for a in objs:
        if isinstance(a, mx.array) and id(a) not in seen:
            seen.add(id(a)); keep.append(a); s += a.nbytes
    return s
gb = nb(vars(m.head).values())
for b in m.blocks:
    t = b.tmix
    lins = ([t._rkv_fused] if t._rkv_fused is not None else [t.r_proj]) + [t.o_proj, b.cmix.key, b.cmix.value]
    for l in lins:
        gb += nb(vars(l).values())

med = {k: float(np.median(v)) for k, v in V.items()}
print("=== %s, K=%d, раундов %d ===" % (os.path.basename(COMP), K, ROUNDS))
for k, v in V.items():
    print("  %-11s %7.3f мс/ток  разброс %4.1f%%" % (k, med[k], 100 * (max(v) - min(v)) / med[k]))
print("  выигрыш «всё, кроме GEMV, бесплатно»: %.3f мс = %.1f%% шага" % (med["full"] - med["gemv_chain"], 100 * (1 - med["gemv_chain"] / med["full"])))
print("  байт в цепочке %.1f МБ -> %.1f ГБ/с = %.0f%% от 104" % (gb / 1e6, gb / 1e6 / med["gemv_chain"], 100 * gb / 1e6 / med["gemv_chain"] / 104))
print("  шаг: 900.9 МБ / full = %.1f ГБ/с = %.0f%% от 104" % (900.9 / med["full"], 100 * 900.9 / med["full"] / 104))
print("  примитивы цепочки:", dict(cnt.most_common(12)))
print("  своп %+.1f МБ" % (swap_used() - sw0))
