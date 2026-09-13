"""Шапка слоя, ход 2: не КЕРНЕЛЬ, а штатный mx.fast.layer_norm.

Замер bench_head_ab 12.09: снятие mean/var из _layer_norm даёт 0.566 мс
(13.662 -> 13.096), сам аффинный хвост бесплатен, token-shift 0.087,
лерпы в остатке ~0.07. То есть вся статья шапки сидит в РЕДУКЦИЯХ
рукописной нормы, а не в запусках и не в лерпах.

_layer_norm у нас: mean -> вычитание -> квадрат -> mean -> rsqrt -> аффин.
В MLX 0.31.2 есть mx.fast.layer_norm -- то же одним примитивом.
Вопрос замера: сколько из потолка 0.566 он берёт.

ЧИСЛА. Порядок редукции другой, бит-в-бит НЕ ожидается. Здесь снимается
relmax логитов и greedy-траектория; решать по ним нельзя (пороги приёмки
10.09: решает KL между плечами), но если relmax уедет за 1e-2 -- дальше
мерить нечего.
"""
import gc, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]
ROUNDS = int(os.environ.get("RWKVQ_AB_ROUNDS", "7"))
REPS = int(os.environ.get("RWKVQ_AB_REPS", "7"))
WARM = 3
MODES = ["off", "lnfast", "lnaff"]

_LN = qm._layer_norm


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    return float(num) * (1024 if unit == "G" else 1)


def _ln_fast(x, weight, bias, eps=1e-5, fast=None):
    return mx.fast.layer_norm(x, weight, bias, eps)


def _ln_aff(x, weight, bias, eps=1e-5, fast=None):
    return x * weight + bias


def setarm(mode):
    qm._layer_norm = {"lnfast": _ln_fast, "lnaff": _ln_aff}.get(mode, _LN)


def measure(m, state, mode):
    setarm(mode)
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    step = m.step

    def full():
        lg, state["st"] = step(state["tok"][None], state["st"])
        state["tok"] = mx.argmax(lg[:, -1], axis=-1)
        return state["tok"]

    for _ in range(WARM):
        mx.eval(full())
    mx.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(full())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def numerics(m, nstep=24, T=512):
    rng = np.random.default_rng(7)
    idxP = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
    out = {}
    for mode in ("off", "lnfast"):
        setarm(mode)
        if hasattr(m, "_step_compiled"):
            del m._step_compiled
        step = m.step
        st = m.init_state(1)
        lg, st = step(idxP, st, True)
        mx.eval(lg)
        pre = np.array(lg[:, -1].astype(mx.float32))
        tok = mx.argmax(lg[:, -1], axis=-1)
        toks = []
        for _ in range(nstep):
            lg, st = step(tok[None], st)
            tok = mx.argmax(lg[:, -1], axis=-1)
            mx.eval(tok)
            toks.append(int(np.array(tok)[0]))
        out[mode] = (pre, toks)
    a, b = out["off"][0], out["lnfast"][0]
    den = np.maximum(np.abs(a).max(), 1e-6)
    print("префилл T=%d: relmax логитов %.3e, max|d| %.3e" % (
        T, np.abs(a - b).max() / den, np.abs(a - b).max()), flush=True)
    ta, tb = out["off"][1], out["lnfast"][1]
    same = sum(1 for i in range(len(ta)) if ta[i] == tb[i])
    print("greedy %d токенов: совпало %d/%d" % (len(ta), same, len(ta)),
          flush=True)
    if ta != tb:
        print("  off   :", ta, flush=True)
        print("  lnfast:", tb, flush=True)


def main():
    sw0 = swap_mb()
    m = qm.QuantRWKV7(load_raw(PATH))
    print("файл: %s" % os.path.basename(PATH), flush=True)
    numerics(m)
    gc.collect()
    mx.clear_cache()
    st = m.init_state(1)
    setarm("off")
    lg, st = m.forward_stateful(
        mx.array(np.arange(1, 65, dtype=np.int32))[None], st, last_only=True)
    mx.eval(lg)
    t0 = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(t0)
    gc.collect()
    mx.clear_cache()
    states = {k: {"st": st, "tok": t0} for k in MODES}
    res = {k: [] for k in MODES}
    for i in range(ROUNDS):
        for k in MODES:
            res[k].append(measure(m, states[k], k))
        print("раунд %d: " % i + "  ".join(
            "%s %.3f" % (k, res[k][-1]) for k in MODES), flush=True)
    qm._layer_norm = _LN
    med = {k: float(np.median(res[k])) for k in MODES}
    print("-" * 64, flush=True)
    for k in MODES:
        print("%-7s %8.3f мс = %6.1f т/с   снято %+7.3f мс" % (
            k, med[k], 1000.0 / med[k], med["off"] - med[k]), flush=True)
    print("доля потолка, взятая fast: %.0f%%" % (
        100.0 * (med["off"] - med["lnfast"]) /
        max(med["off"] - med["lnaff"], 1e-9)), flush=True)
    for k in MODES:
        print("разброс %-7s %.3f..%.3f" % (k, min(res[k]), max(res[k])),
              flush=True)
    print("своп дельта %+.1f МБ" % (swap_mb() - sw0), flush=True)


main()
