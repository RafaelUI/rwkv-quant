"""ПРИОРИТЕТ 2: премисса шапки слоя. Сколько из 0.726 мс -- НЕ арифметика.

Статья 10.09 "нормы + сдвиги + лерпы" = 0.726 мс при дне по памяти ~0.01.
11.09 на пред-WKV выяснилось, что mx.compile уже сводит такую цепочку
примерно к одному диспатчу, и 0.282 из 0.574 -- цена САМОГО запуска.
Здесь тот же вопрос для шапки, и ДО написания ядра.

МЕТОД. Плечи чередуются в ОДНОМ процессе (закон 25), перед каждым
сбрасывается кеш компиляции (иначе померится одно плечо дважды).
Заглушки ОБЯЗАНЫ держать входы живыми (закон 11.09): ни одна не
обнуляет x, иначе compile выкинет ветки как мёртвые и плечо померит
чужую статью.

ПЛЕЧИ:
  off    -- настоящий шаг;
  lnaff  -- layer_norm -> x*w+b: снимает ДВЕ РЕДУКЦИИ (mean, var), 50 вызовов;
  lnid   -- layer_norm -> x: снимает норму целиком;
  ts     -- token_shift_stateful -> (x, x): снимает concat/сдвиг, 48 вызовов;
  both   -- lnaff + ts.
ЧИСЛА ПЛЕЧ ЗАВЕДОМО НЕВЕРНЫ, это потолки, а не варианты правки.
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
MODES = ["off", "lnaff", "lnid", "ts", "both"]

_LN = qm._layer_norm
_TS = qm._token_shift_stateful


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    return float(num) * (1024 if unit == "G" else 1)


def _ln_aff(x, weight, bias, eps=1e-5, fast=None):
    return x * weight + bias


def _ln_id(x, weight, bias, eps=1e-5, fast=None):
    return x


def _ts_id(x, prev):
    return x, x


def setarm(mode):
    qm._layer_norm = {"lnaff": _ln_aff, "lnid": _ln_id,
                      "both": _ln_aff}.get(mode, _LN)
    qm._token_shift_stateful = _ts_id if mode in ("ts", "both") else _TS


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


def main():
    sw0 = swap_mb()
    m = qm.QuantRWKV7(load_raw(PATH))
    print("файл: %s" % os.path.basename(PATH), flush=True)
    st = m.init_state(1)
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
    qm._token_shift_stateful = _TS
    med = {k: float(np.median(res[k])) for k in MODES}
    print("-" * 64, flush=True)
    for k in MODES:
        print("%-6s %8.3f мс = %6.1f т/с   дельта к off %+7.3f мс" % (
            k, med[k], 1000.0 / med[k], med["off"] - med[k]), flush=True)
    print("разброс off %.3f..%.3f" % (min(res["off"]), max(res["off"])),
          flush=True)
    print("своп дельта %+.1f МБ" % (swap_mb() - sw0), flush=True)


main()
