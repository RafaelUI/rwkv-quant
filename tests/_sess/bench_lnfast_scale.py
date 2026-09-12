"""mx.fast.layer_norm: префилл и второй масштаб (закон 10).

Декод на 1.5B уже померен: 13.649 -> 13.218 мс, 0.431 из потолка 0.497.
Здесь два вопроса, которые из декода не следуют: (1) берёт ли примитив
что-нибудь на ПРЕФИЛЛЕ, где редукция идёт по 512 строкам, а не по одной;
(2) держится ли доля на другом масштабе -- у 0.1B 12 слоёв и D=768.
Плечи чередуются в одном процессе, кеш компиляции сбрасывается перед
каждым (иначе смена функции после трассировки не меняет ничего).
"""
import gc, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]
ROUNDS = int(os.environ.get("RWKVQ_AB_ROUNDS", "5"))
REPS = int(os.environ.get("RWKVQ_AB_REPS", "5"))
TS = [int(x) for x in os.environ.get("RWKVQ_TS", "512").split(",")]
WARM = 2
MODES = ["off", "lnfast"]
_LN = qm._layer_norm


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    return float(num) * (1024 if unit == "G" else 1)


def _ln_fast(x, weight, bias, eps=1e-5):
    return mx.fast.layer_norm(x, weight, bias, eps)


def setarm(mode):
    qm._layer_norm = _ln_fast if mode == "lnfast" else _LN


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


def prefill_arm(m, mode, idx):
    setarm(mode)
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    step = m.step

    def pf():
        st = m.init_state(1)
        lg, _ = step(idx, st, True)
        return lg
    return bench(pf)


def decode_arm(m, mode, state):
    setarm(mode)
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    step = m.step

    def one():
        lg, state["st"] = step(state["tok"][None], state["st"])
        state["tok"] = mx.argmax(lg[:, -1], axis=-1)
        return state["tok"]
    return bench(one)


def main():
    sw0 = swap_mb()
    m = qm.QuantRWKV7(load_raw(PATH))
    print("файл: %s, слоёв %d" % (os.path.basename(PATH), m.n_layer),
          flush=True)
    rng = np.random.default_rng(0)
    for T in TS:
        idx = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
        r = {k: [] for k in MODES}
        for i in range(ROUNDS):
            for k in MODES:
                r[k].append(prefill_arm(m, k, idx))
        med = {k: float(np.median(r[k])) for k in MODES}
        print("ПРЕФИЛЛ T=%4d: off %8.1f мс (%6.1f т/с) | fast %8.1f мс "
              "(%6.1f т/с) | снято %+6.1f мс = %+.1f%%" % (
                  T, med["off"], 1000.0 * T / med["off"], med["lnfast"],
                  1000.0 * T / med["lnfast"], med["off"] - med["lnfast"],
                  100.0 * (med["off"] - med["lnfast"]) / med["off"]),
              flush=True)
        print("   разброс off %.1f..%.1f, fast %.1f..%.1f" % (
            min(r["off"]), max(r["off"]), min(r["lnfast"]),
            max(r["lnfast"])), flush=True)
        gc.collect()
        mx.clear_cache()
    setarm("off")
    st = m.init_state(1)
    lg, st = m.forward_stateful(
        mx.array(np.arange(1, 65, dtype=np.int32))[None], st, last_only=True)
    mx.eval(lg)
    t0 = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(t0)
    gc.collect()
    mx.clear_cache()
    states = {k: {"st": st, "tok": t0} for k in MODES}
    r = {k: [] for k in MODES}
    for i in range(ROUNDS):
        for k in MODES:
            r[k].append(decode_arm(m, k, states[k]))
    med = {k: float(np.median(r[k])) for k in MODES}
    print("ДЕКОД B=1: off %.3f мс (%.1f т/с) | fast %.3f мс (%.1f т/с) | "
          "снято %+.3f мс = %+.1f%%" % (
              med["off"], 1000.0 / med["off"], med["lnfast"],
              1000.0 / med["lnfast"], med["off"] - med["lnfast"],
              100.0 * (med["off"] - med["lnfast"]) / med["off"]), flush=True)
    print("   разброс off %.3f..%.3f, fast %.3f..%.3f" % (
        min(r["off"]), max(r["off"]), min(r["lnfast"]), max(r["lnfast"])),
          flush=True)
    qm._layer_norm = _LN
    print("своп дельта %+.1f МБ" % (swap_mb() - sw0), flush=True)


main()
