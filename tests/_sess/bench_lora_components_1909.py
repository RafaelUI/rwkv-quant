"""LoRA ПО КОМПОНЕНТАМ (19.09): квантованные ветки w/a/v/g (LORA_Q="sep", 8 бит,
gs 64 вниз / 32 вверх) на настоящих весах, под mx.compile, как в шаге.
  <ветка>      -- down -> нелинейность -> up зависимой цепочкой по слоям;
  <ветка>_down -- только down (2048 -> ранг), зависимой цепочкой;
  <ветка>_up   -- только up (ранг -> 2048), зависимой цепочкой;
  lora_layer   -- весь QuantTMix._lora слоя (4 ветки, как в шаге), слои цепочкой;
  lora_indep   -- то же без зависимостей между слоями;
  addchain     -- цена добавочной операции x0 + 0*dep на уровень (вычитается).
Идеал зависимого читателя -- 17 мкс + байты/107 ГБ/с на запуск.
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
TM = [b.tmix for b in m.blocks]
D = int(m.emb_weight.shape[1])
GD, GU, QB = qm.LORA_GS_DOWN, qm.LORA_GS_UP, qm.LORA_QBITS
rng = np.random.default_rng(0)
x0 = mx.array(rng.standard_normal((1, 1, D)).astype(np.float32) * 0.1); mx.eval(x0)


def qmm(x, tr, gs):
    wq, sc, bi = tr
    return mx.quantized_matmul(x.astype(mx.float16), wq, scales=sc, biases=bi,
                               transpose=True, group_size=gs, bits=QB)


def layers_with(nm):
    return [t for t in TM if nm in t._lq_names]


def branch(nm, part):
    ts = layers_with(nm)
    def f(x):
        dep = None
        for t in ts:
            i = t._lq_names.index(nm)
            xin = x if dep is None else x + dep * 0.0
            if part == "up":
                r = t._lq_B[i][0].shape[1] * 32 // QB
                h = xin.reshape(-1)[:r].reshape(1, 1, r)
                y = qmm(h, t._lq_B[i], GU).astype(mx.float32)
            else:
                h = qmm(xin, t._lq_A[i], GD).astype(mx.float32)
                if part == "down":
                    y = h
                else:
                    h = mx.tanh(h) if nm == "w" else (mx.sigmoid(h) if nm == "g" else h)
                    y = qmm(h, t._lq_B[i], GU).astype(mx.float32)
            dep = y.reshape(-1)[:1]
        return dep
    return mx.compile(f), len(ts)


def lora_layer(dependent):
    def f(x):
        dep, acc = None, None
        for t in TM:
            xin = x if (dep is None or not dependent) else x + dep * 0.0
            outs = t._lora(xin, xin, xin, xin, xin, xin)
            s = sum(o.reshape(-1)[:1].astype(mx.float32) for o in outs if o is not None)
            if dependent:
                dep = s
            else:
                acc = s if acc is None else acc + s
        return dep if dependent else acc
    return mx.compile(f), len(TM)


def addchain():
    def f(x):
        dep = None
        for _ in TM:
            xin = x if dep is None else x + dep * 0.0
            dep = xin.reshape(-1)[:1] * 1.0001
        return dep
    return mx.compile(f), len(TM)


FN = {}
for nm in ("w", "a", "v", "g"):
    FN[nm] = branch(nm, "full"); FN[nm + "_down"] = branch(nm, "down"); FN[nm + "_up"] = branch(nm, "up")
FN["lora_layer"] = lora_layer(True); FN["lora_indep"] = lora_layer(False); FN["addchain"] = addchain()


def t(k, reps=7):
    f = FN[k][0]
    mx.eval(f(x0)); mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize(); a = time.perf_counter(); mx.eval(f(x0)); mx.synchronize()
        ts.append(time.perf_counter() - a)
    return float(np.median(ts)) * 1e6


def nb(tr):
    return sum(a.nbytes for a in tr)


def swap():
    return float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))


sw0 = swap()
keys = list(FN)
V = {k: [] for k in keys}
for rd in range(ROUNDS):
    for k in (keys if rd % 2 == 0 else keys[::-1]):
        V[k].append(t(k))
med = {k: np.median(V[k]) / FN[k][1] for k in keys}
add = med["addchain"]
t1 = TM[1]
print("=== %s ===" % os.path.basename(COMP))
print("добавочная операция на уровень (вычитается): %.1f мкс" % add)
print("ветка | ранг | КБ down/up | down мкс | up мкс | down+нелин+up мкс | ГБ/с ветки | идеал 2 запуска мкс | мс/токен")
tot = 0
for nm in ("w", "a", "v", "g"):
    i = t1._lq_names.index(nm)
    bd, bu = nb(t1._lq_A[i]), nb(t1._lq_B[i])
    r = t1._lq_A[i][0].shape[0]
    dn, up, fu = med[nm + "_down"] - add, med[nm + "_up"] - add, med[nm] - add
    ideal = 2 * 17 + (bd + bu) / 107e3
    n = FN[nm][1]
    tot += fu * n / 1e3
    print("%-5s | %4d | %5.0f / %5.0f | %6.1f | %6.1f | %6.1f | %5.1f | %5.1f | %.3f" % (
        nm, r, bd / 1e3, bu / 1e3, dn, up, fu, (bd + bu) / fu / 1e3, ideal, fu * n / 1e3))
print("сумма веток по цепочке (последовательно): %.3f мс/токен" % tot)
lb = sum(nb(tr) for t in TM for tr in (t._lq_A + t._lq_B))
print("весь _lora слоя цепочкой: %.1f мкс/слой = %.3f мс/токен; без зависимостей: %.1f мкс/слой; байт LoRA на токен %.1f МБ -> %.1f ГБ/с в цепочке" % (
    med["lora_layer"] - add, (med["lora_layer"] - add) * len(TM) / 1e3, med["lora_indep"], lb / 1e6, lb / 1e6 / ((med["lora_layer"] - add) * len(TM) / 1e3)))
for k in keys:
    print("  %-11s разброс %4.1f%%" % (k, 100 * (max(V[k]) - min(V[k])) / np.median(V[k])))
print("своп %+.1f МБ" % (swap() - sw0))
