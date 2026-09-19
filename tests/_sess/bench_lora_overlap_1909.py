"""ПЕРЕКРЫВАЕТСЯ ЛИ LoRA С ФЬЮЗОМ rkv (19.09). В шаге обе ветви зависят только
от лерпов слоя и друг от друга не зависят. Цепочка по 24 слоям, вход слоя =
x0 + 0 * (сумма первых элементов выходов прошлого слоя):
  rkv   -- только фьюз r/k/v;   lora -- только _lora (4 ветки);
  both  -- оба от одного входа, следующий слой ждёт оба.
both ~ rkv + lora -- перекрытия нет, LoRA целиком на критическом пути;
both ~ max(rkv, lora) -- перекрываются.
"""
import os, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw
m = qm.QuantRWKV7(load_raw("/Users/s/Develop/WKV-kvant/" + os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")))
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)
TM = [b.tmix for b in m.blocks]; D = int(m.emb_weight.shape[1])
x0 = mx.array(np.random.default_rng(0).standard_normal((1, 1, D)).astype(np.float32) * 0.1); mx.eval(x0)


def mk(use_rkv, use_lora):
    def f(x):
        dep = None
        for t in TM:
            xin = x if dep is None else x + dep * 0.0
            s = None
            if use_rkv:
                o = t._rkv_fused(mx.concatenate([xin.reshape(1, D)] * 3, axis=0))
                s = o.reshape(-1)[:1]
            if use_lora:
                outs = t._lora(xin, xin, xin, xin, xin, xin)
                sl = sum(q.reshape(-1)[:1].astype(mx.float32) for q in outs if q is not None)
                s = sl if s is None else s + sl
            dep = s
        return dep
    return mx.compile(f)


FN = {"rkv": mk(True, False), "lora": mk(False, True), "both": mk(True, True)}


def t(k, reps=7):
    f = FN[k]; mx.eval(f(x0)); mx.synchronize(); ts = []
    for _ in range(reps):
        mx.synchronize(); a = time.perf_counter(); mx.eval(f(x0)); mx.synchronize(); ts.append(time.perf_counter() - a)
    return float(np.median(ts)) * 1e6 / len(TM)


V = {k: [] for k in FN}
for rd in range(11):
    for k in (list(FN) if rd % 2 == 0 else list(FN)[::-1]):
        V[k].append(t(k))
md = {k: np.median(v) for k, v in V.items()}
for k in FN:
    print("  %-5s %6.1f мкс/слой  разброс %4.1f%%" % (k, md[k], 100 * (max(V[k]) - min(V[k])) / md[k]))
print("  rkv + lora = %.1f; max = %.1f; both = %.1f -> перекрыто %.0f%% LoRA" % (
    md["rkv"] + md["lora"], max(md["rkv"], md["lora"]), md["both"], 100 * (md["rkv"] + md["lora"] - md["both"]) / md["lora"]))
