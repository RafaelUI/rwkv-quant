"""ПРЕМИССА «LoRA-down В ЗАПУСКЕ ФЬЮЗА rkv» (19.09). К фьюзу r/k/v слоя
приклеиваются E лишних строк ТОГО ЖЕ формата (первые E строк o_proj того же
слоя) с ЧЕТВЁРТЫМ входом, как если бы это были LoRA-down (вес LoRA-down слоя
1.11 МБ; строка proj ~1.18 КБ -> E=1024 ~ 1.21 МБ). Кернель тот же
(_get_kernel_k3, out_per=2048: вход строки = row/OUT_PER). Вопрос: сколько
стоят лишние ~1.1 МБ ВНУТРИ уже идущего запуска -- ~11 мкс (чистые байты при
полной полосе) или ~34 (сколько LoRA сейчас добавляет поверх rkv).
Плечи (зависимая цепочка по 24 слоям): rkv (штатный фьюз), rkv_E0 (тот же
кернель прямым вызовом -- сверка прибора), rkv_E512, rkv_E1024. Первые 6144
выходов плеч с E обязаны совпасть со штатным бит-в-бит.
"""
import os, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.formats.reader import load_raw
m = qm.QuantRWKV7(load_raw("/Users/s/Develop/WKV-kvant/" + os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")))
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)
TM = [b.tmix for b in m.blocks]
f0 = TM[1]._rkv_fused
assert f0._k3 and not gw.K3_XSUM and not gw.K3_HALF, (f0._k3, gw.K3_XSUM, gw.K3_HALF)
IN, OP, NB = f0.in_features, f0.out_per, f0.NB
NSG, RS = gw._k3_cfg(IN, OP, f0.xbits)


def rows(a, OUT, E):
    return a.reshape(OUT, -1)[:E].reshape((-1,) + a.shape[1:]) if a.ndim == 1 else a[:E]


BUF = {}
for E in (0, 512, 1024):
    L = []
    for t in TM:
        f, o = t._rkv_fused, t.o_proj
        if E:
            parts = [mx.concatenate([getattr(f, n), rows(getattr(o, n), o.out_features, E)], axis=0) for n in ("qblk", "qsqm", "ddm")]
        else:
            parts = [f.qblk, f.qsqm, f.ddm]
        L.append(parts)
    mx.eval([a for p in L for a in p])
    BUF[E] = L


def direct(E, i, xs):
    OUT = 3 * OP + E
    K = 3 + (1 if E else 0)
    kern = gw._get_kernel_k3(IN, OUT, f0.xbits, NSG, RS, out_per=OP, xin=False)
    xbsum = mx.sum(xs.reshape(K, NB, 32), axis=2)
    q, s, d = BUF[E][i]
    return kern(inputs=[xs, q, s, d, xbsum], grid=(OUT // (NSG * RS) * NSG * 32, 1, 1),
                threadgroup=(NSG * 32, 1, 1), output_shapes=[(1, OUT)], output_dtypes=[mx.float32])[0]


x3 = mx.array(np.random.default_rng(0).standard_normal((3, IN)).astype(np.float32) * 0.1)
x4 = mx.concatenate([x3, x3[:1] * 0.5], axis=0); mx.eval(x3, x4)


def mk(E):
    def f(x3, x4):
        dep = None
        for i, t in enumerate(TM):
            if E is None:
                xin = x3 if dep is None else x3 + dep * 0.0
                o = t._rkv_fused(xin)
            else:
                src = x4 if E else x3
                xin = src if dep is None else src + dep * 0.0
                o = direct(E, i, xin)
            dep = o.reshape(-1)[:1]
        return dep
    return mx.compile(f)


# сверка бит-в-бит на одном слое
ref = TM[5]._rkv_fused(x3).reshape(-1)
for E in (0, 512, 1024):
    got = direct(E, 5, x4 if E else x3).reshape(-1)[:3 * OP]
    print("  сверка E=%d: первые %d выходов %s" % (E, 3 * OP, "БИТ-В-БИТ" if mx.array_equal(ref, got).item() else "РАСХОДЯТСЯ"))
EB = {E: sum(a.nbytes for a in BUF[E][5]) / 1e6 for E in (0, 512, 1024)}
FN = {"rkv": mk(None), "rkv_E0": mk(0), "rkv_E512": mk(512), "rkv_E1024": mk(1024)}


def t(k, reps=7):
    f = FN[k]; mx.eval(f(x3, x4)); mx.synchronize(); ts = []
    for _ in range(reps):
        mx.synchronize(); a = time.perf_counter(); mx.eval(f(x3, x4)); mx.synchronize(); ts.append(time.perf_counter() - a)
    return float(np.median(ts)) * 1e6 / len(TM)


V = {k: [] for k in FN}
for rd in range(int(os.environ.get("RWKVQ_ROUNDS", "11"))):
    for k in (list(FN) if rd % 2 == 0 else list(FN)[::-1]):
        V[k].append(t(k))
md = {k: np.median(v) for k, v in V.items()}
for k in FN:
    print("  %-10s %6.1f мкс/слой  разброс %4.1f%%" % (k, md[k], 100 * (max(V[k]) - min(V[k])) / md[k]))
for E in (512, 1024):
    dmb = EB[E] - EB[0]
    print("  E=%d: +%.2f МБ -> +%.1f мкс (чистые байты при 104 ГБ/с: %.1f мкс); на токен +%.2f мс" % (
        E, dmb, md["rkv_E%d" % E] - md["rkv_E0"], dmb / 104e-3, (md["rkv_E%d" % E] - md["rkv_E0"]) * len(TM) / 1e3))
