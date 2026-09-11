"""Свип NB_CHUNK на батчевом декоде: сколько раз веса едут из DRAM.

При N < GEMM_MIN_BATCH_NB=128 батч режется на чанки по NB_CHUNK=4, и
веса перечитываются ceil(N/NB_CHUNK) раз. Оптимум 4 снят 19.07 на
префилльных формах; здесь проверяется батчевый декод, где N мал и
форма другая. Числа сверяются с чанком 4 -- кернель обещает бит-в-бит
по паре (строка, колонка), и это проверяется, а не принимается."""
import gc, os, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]
B = int(os.environ.get("RWKVQ_B", "16"))
REPS, WARM = 7, 3
rng = np.random.default_rng(0)
m = qm.QuantRWKV7(load_raw(PATH))
print("файл: %s, B=%d, GEMM_MIN_BATCH_NB=%d" % (
    os.path.basename(PATH), B, gw.GEMM_MIN_BATCH_NB), flush=True)

st0 = m.init_state(B)
idx = mx.array(rng.integers(1, 60000, size=(B, 64)).astype(np.int32))
lg, st0 = m.forward_stateful(idx, st0, last_only=True)
mx.eval(lg)
tok0 = mx.argmax(lg[:, -1], axis=-1).reshape(B, 1)
mx.eval(tok0)

ref = None
for chunk in (4, 8, 16, 32):
    gw.NB_CHUNK = chunk
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    stp = m.step
    try:
        out, _ = stp(tok0, st0)
        mx.eval(out)
        a = np.array(out.astype(mx.float32))
        if ref is None:
            ref, same = a, "эталон"
        else:
            same = "бит-в-бит" if np.array_equal(a, ref) else (
                "РАСХОЖДЕНИЕ max %.3e" % float(np.max(np.abs(a - ref))))
        state = {"st": st0, "tok": tok0}
        def one():
            l2, state["st"] = stp(state["tok"], state["st"])
            state["tok"] = mx.argmax(l2[:, -1], axis=-1).reshape(B, 1)
            return state["tok"]
        for _ in range(WARM):
            mx.eval(one())
        mx.synchronize()
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            mx.eval(one())
            mx.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ms = float(np.median(ts))
        print("NB_CHUNK=%2d: %8.3f мс/шаг, %7.1f ток/с  [%s]" % (
            chunk, ms, 1000.0 * B / ms, same), flush=True)
    except Exception as e:
        print("NB_CHUNK=%2d: ОШИБКА %s" % (chunk, str(e)[:120]), flush=True)
    gc.collect(); mx.clear_cache()
print("ГОТОВО", flush=True)
