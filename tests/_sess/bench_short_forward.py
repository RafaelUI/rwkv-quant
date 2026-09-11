"""Цена короткого форварда против цепочки декод-шагов.

ЗАЧЕМ. Декод одного потока нельзя распараллелить по токенам: состояние
RWKV -- строгая последовательная зависимость. Единственный способ
обогнать её на ОДНОМ потоке -- спекулятивный декод: черновик маленькой
моделью, проверка большой ОДНИМ форвардом на T токенов. Выигрыш есть
ровно тогда, когда форвард на T дешевле T шагов. Здесь это меряется.
"""
import gc, os, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

REPS, WARM = 7, 3
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


for path in sys.argv[1:]:
    m = qm.QuantRWKV7(load_raw(path))
    print("=== %s ===" % os.path.basename(path), flush=True)
    one = None
    for T in (1, 2, 4, 8, 16, 32, 64):
        idx = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
        if hasattr(m, "_step_compiled"):
            del m._step_compiled
        stp = m.step
        def fw():
            st = m.init_state(1)
            lg, _ = stp(idx, st, True)
            return lg
        ms = bench(fw)
        if one is None:
            one = ms
        print("T=%3d: %8.3f мс, %6.2f шага по цене, %7.1f ток/с" % (
            T, ms, ms / one, 1000.0 * T / ms), flush=True)
        gc.collect(); mx.clear_cache()
    del m
    gc.collect(); mx.clear_cache()
print("ГОТОВО", flush=True)
