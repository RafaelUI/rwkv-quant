"""Масштабирование декода по батчу и текущее состояние префилла.

ЗАЧЕМ. Декод одного потока упирается в чтение весов: 900.9 МБ трафика на
токен при пропускной 104 ГБ/с дают пол 8.89 мс. Веса читаются целиком
НЕЗАВИСИМО от того, сколько последовательностей считается разом, поэтому
батч -- единственная ось, где трафик амортизируется. Здесь это проверяется,
а не предполагается.
"""
import gc, os, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]
REPS, WARM = 5, 3
BS = [int(x) for x in os.environ.get("RWKVQ_BS", "1,2,4,8,16").split(",")]

m = qm.QuantRWKV7(load_raw(PATH))
print("файл: %s" % os.path.basename(PATH), flush=True)
rng = np.random.default_rng(0)


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


print("--- ПРЕФИЛЛ, B=1 ---", flush=True)
for T in (256, 512, 1024):
    idx = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    pstep = m.step          # КОМПИЛИРОВАННЫЙ путь: forward_stateful
                            # напрямую -- это не префилл модели, а
                            # его некомпилированный двойник, он
                            # медленнее на треть (записано 16.08).
    def pf():
        st = m.init_state(1)
        lg, _ = pstep(idx, st, True)
        return lg
    ms = bench(pf)
    print("T=%4d: %8.1f мс = %6.1f ток/с" % (T, ms, 1000.0 * T / ms), flush=True)
    gc.collect(); mx.clear_cache()

print("--- ДЕКОД ПО БАТЧУ ---", flush=True)
base = None
for B in BS:
    try:
        st = m.init_state(B)
        idx = mx.array(rng.integers(1, 60000, size=(B, 64)).astype(np.int32))
        lg, st = m.forward_stateful(idx, st, last_only=True)
        mx.eval(lg)
        tok = mx.argmax(lg[:, -1], axis=-1).reshape(B, 1)
        mx.eval(tok)
        if hasattr(m, "_step_compiled"):
            del m._step_compiled
        step = m.step
        state = {"st": st, "tok": tok}

        def one():
            l2, state["st"] = step(state["tok"], state["st"])
            state["tok"] = mx.argmax(l2[:, -1], axis=-1).reshape(B, 1)
            return state["tok"]

        ms = bench(one)
        tps = 1000.0 * B / ms
        if base is None:
            base = tps
        print("B=%2d: %7.3f мс/шаг, %7.1f ток/с, x%.2f к B=1, "
              "на поток %5.1f ток/с" % (
                  B, ms, tps, tps / base, 1000.0 / ms), flush=True)
        del st, state
        gc.collect(); mx.clear_cache()
    except Exception as e:
        print("B=%2d: ОШИБКА %s" % (B, str(e)[:100]), flush=True)
        break
print("ГОТОВО", flush=True)
