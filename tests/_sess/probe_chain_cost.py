"""Честный состав шага (через ФЬЮЗНУТЫЙ путь) и цена последовательного запуска.

Дыра, найденная 09.09: bench_step_decompose считает GEMV-фазу раздельными
r_proj/k_proj/v_proj, а настоящий шаг идёт _rkv_fused (один launch вместо трёх).
Поэтому деление 78/22 неверно в обе стороны. Здесь GEMV-прокси собран из тех же
ядер, что реально работают в декоде.

Вторая часть: та же работа цепочкой против той же работы независимо. Байты
одинаковы, разница есть чистая цена последовательности на N запусков.
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")

import mlx.core as mx
import numpy as np
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.formats.reader import load_raw
from rwkv_metal.kernel.wkv7 import wkv7_infer
from rwkv_quant.backends.metal.fused_tail import wkv_tail
from rwkv_quant.backends.metal.quant_model import _layer_norm

PATH = sys.argv[1]
ROUNDS, REPS, WARM = 9, 7, 3


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    return float(num) * (1024 if unit == "G" else 1)


def bench(fn):
    for _ in range(WARM):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    m = QuantRWKV7(load_raw(PATH))
    D = int(m.emb_weight.shape[1])
    st = m.init_state(1)
    tok = mx.array(np.array([1], dtype=np.int32))
    logits, st2 = m.step(tok[None], st)
    mx.eval(logits)
    nf = sum(1 for b in m.blocks if getattr(b.tmix, "_rkv_fused", None) is not None)
    print("файл: %s" % os.path.basename(PATH), flush=True)
    print("слоёв с живым _rkv_fused: %d из %d" % (nf, len(m.blocks)), flush=True)
    gc.collect()
    mx.clear_cache()

    x = mx.array(np.random.randn(1, D).astype(np.float32))
    mx.eval(x)
    state = {"st": st, "tok": tok}

    def full():
        lg, state["st"] = m.step(state["tok"][None], state["st"])
        state["tok"] = mx.argmax(lg[:, -1], axis=-1)
        return state["tok"]

    def gemv_fused():
        y = x
        for b in m.blocks:
            tm, cm = b.tmix, b.cmix
            f = getattr(tm, "_rkv_fused", None)
            if f is not None:
                rkv = f(mx.concatenate([y, y, y], axis=0))
                y = tm.o_proj(rkv[0].reshape(1, D))
            else:
                y = tm.o_proj((tm.r_proj(y) + tm.k_proj(y) + tm.v_proj(y)) * 0.33)
            y = cm.value(cm.key(y))
        return m.head(y)

    def gemv_sep():
        y = x
        for b in m.blocks:
            tm, cm = b.tmix, b.cmix
            y = (tm.r_proj(y) + tm.k_proj(y) + tm.v_proj(y) + tm.o_proj(y)) * 0.25
            y = cm.value(cm.key(y))
        return m.head(y)

    def chain(n):
        def run():
            y = x
            for i in range(n):
                y = m.blocks[i].tmix.r_proj(y)
            return y
        return run

    def indep(n):
        def run():
            return [m.blocks[i].tmix.r_proj(x) for i in range(n)]
        return run

    # WKV чередуется ЗДЕСЬ ЖЕ: отдельным процессом он мерится под другой
    # тепловой историей (09.09: пол eval 0.221 против 0.388 мс в двух
    # прогонах одного скрипта), и доля от шага получается фиктивной.
    Hn, Sn = m.n_head, m.head_size
    rng = np.random.default_rng(0)
    def t4():
        return mx.array(rng.standard_normal((1, 1, Hn, Sn)).astype(np.float32))
    wargs = [[t4() for _ in range(5)] for _ in range(24)]
    wws = [mx.array((-0.5 - 0.5 * np.abs(rng.standard_normal((1, 1, Hn, Sn)))).astype(np.float32))
           for _ in range(24)]
    whs = [mx.array((rng.standard_normal((1, Hn, Sn, Sn)) * 0.01).astype(np.float32))
           for _ in range(24)]
    for _a in wargs:
        mx.eval(_a)
    mx.eval(wws, whs)

    def wkv_chain():
        h = whs[0]
        out = None
        for i in range(24):
            r, k, v, a, b = wargs[i]
            out, h = wkv7_infer(r, wws[i], k, v, a, b, h)
        return [out, h]

    cases = {"full (настоящий шаг)": full, "gemv ФЬЮЗНУТЫЙ": gemv_fused,
             "gemv раздельный": gemv_sep, "пол eval": lambda: x + 1.0}
    x3 = x.reshape(1, 1, D)
    def lora_all():
        # ветки LoRA всех 24 слоёв: 4 down + 4 up на слой при LORA_Q=sep
        res = []
        for b in m.blocks:
            res.extend([t for t in b.tmix._lora(x3, x3, x3, x3, x3, x3)
                        if t is not None])
        return res
    cases["wkv x24 цепочкой"] = wkv_chain
    # Пред-WKV блок как ПРОКСИ: та же арифметика, что в
    # _forward_stateful_fused между LoRA и _wkv_stateful, под mx.compile.
    # Это НИЖНЯЯ ГРАНИЦА цены: здесь 24 слоя идут подряд и compile фьюзит
    # их в одну цепочку, а в настоящем шаге между ними стоят GEMV и LoRA,
    # которые цепочку рвут.
    def _bt():
        return mx.array(rng.standard_normal((1, 1, Hn, Sn)).astype(np.float32))
    flat = []
    for _ in range(24):
        flat.extend([_bt() for _ in range(6)])
    flat.extend([_bt() for _ in range(5)])
    mx.eval(flat)

    def _prewkv(arr):
        wb, ab, vb, kkc, kac = arr[144:149]
        res = []
        for i in range(24):
            yw, ya, yv, k, v, vf = arr[6 * i:6 * i + 6]
            w = mx.exp(-0.606531 * mx.sigmoid(yw + wb))
            a = mx.sigmoid(ya + ab)
            kk = k * kkc
            kk = kk / mx.sqrt((kk * kk).sum(axis=-1, keepdims=True) + 1e-12)
            k2 = k * (1.0 + (a - 1.0) * kac)
            vv = mx.sigmoid(yv + vb)
            v2 = v + (vf - v) * vv
            res.extend([w, k2, v2, -kk, kk * a])
        return res

    _pbc = mx.compile(_prewkv)
    cases["lora x24 (8 запусков/слой)"] = lora_all
    cases["предWKV блок x24"] = lambda: _pbc(flat)

    # Хвост TMix: уже фьюзнут в одно ядро, но запусков всё равно 24.
    tl = [[_bt() for _ in range(5)] for _ in range(24)]
    rk1, lnw1, lnb1 = _bt(), _bt(), _bt()
    for _t in tl:
        mx.eval(_t)
    mx.eval(rk1, lnw1, lnb1)

    def _tails():
        return [wkv_tail(t[0], t[1], t[2], t[3], rk1, lnw1, lnb1, t[4], Hn, Sn)
                for t in tl]
    cases["wkv_tail x24"] = _tails

    # Нормы, сдвиги, лерпы и резидуалы: 48 layer_norm по D на шаг плюс
    # бродкаст [6, D] на слой. Прокси, но геометрия и число редукций те же.
    def _bd():
        return mx.array(rng.standard_normal((1, 1, D)).astype(np.float32))
    nrm = [_bd() for _ in range(72)]
    nv = [mx.array(rng.standard_normal((D,)).astype(np.float32)) for _ in range(4)]
    xcf = mx.array(rng.standard_normal((6, 1, 1, D)).astype(np.float32))
    mx.eval(nrm, nv, xcf)

    def _norms(arr):
        w1, b1, w2, b2 = arr[72:76]
        xc = arr[76]
        y = arr[0]
        res = []
        for i in range(24):
            h = _layer_norm(y, w1, b1)
            xx = arr[24 + i] - h
            xs = h[None] + xx[None] * xc
            y = y + xs[0]
            h2 = _layer_norm(y, w2, b2)
            xx2 = arr[48 + i] - h2
            y = y + h2 * 0.5 + xx2 * 0.5
            res.append(y)
        return res
    _nc = mx.compile(_norms)
    nargs = nrm + nv + [xcf]
    cases["нормы+сдвиги+лерпы x24"] = lambda: _nc(nargs)
    for n in (6, 12, 24):
        cases["цепочка x%d" % n] = chain(n)
        cases["независимо x%d" % n] = indep(n)

    sw0 = swap_mb()
    acc = {k: [] for k in cases}
    for _ in range(ROUNDS):
        for k, fn in cases.items():
            acc[k].append(bench(fn))
    sw1 = swap_mb()
    print("своп %.0f -> %.0f МБ%s" % (sw0, sw1,
          "  *** РОС (закон 11) ***" if sw1 > sw0 + 0.5 else "  (не рос)"), flush=True)
    med = {}
    print("", flush=True)
    print("%-22s %8s %8s" % ("что", "мс", "разброс"), flush=True)
    for k in cases:
        med[k] = float(np.median(acc[k]))
        sp = 100 * (max(acc[k]) - min(acc[k])) / med[k]
        print("%-22s %8.3f %7.1f%%" % (k, med[k], sp), flush=True)
    print("", flush=True)
    for name in ("gemv ФЬЮЗНУТЫЙ", "gemv раздельный"):
        rest = med["full (настоящий шаг)"] - med[name]
        print("по прокси %-16s: GEMV %.3f мс (%.0f%%), остальное %.3f мс (%.0f%%)"
              % (name, med[name], 100 * med[name] / med["full (настоящий шаг)"],
                 rest, 100 * rest / med["full (настоящий шаг)"]), flush=True)
    print("", flush=True)
    print("цена последовательности (цепочка минус независимо):", flush=True)
    for n in (6, 12, 24):
        d = med["цепочка x%d" % n] - med["независимо x%d" % n]
        print("  N=%2d: %7.3f мс на цепочку = %6.1f мкс на запуск" % (n, d, d * 1000 / n), flush=True)


if __name__ == "__main__":
    main()
