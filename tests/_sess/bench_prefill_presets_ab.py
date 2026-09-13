"""ПЕРВЫЙ ЗАМЕР приоритета 1: compression против reduction на префилле,
чередованием плеч в ОДНОМ процессе (законы 1 и 25).

12.09 отношение 1.20x (807.0 против 670.1 мс на T=512) снято в ДВУХ
процессах и утверждением не является. Здесь оба файла живут в одном
процессе (~2.8 ГБ резидентно), плечи чередуются по раундам, своп мерится
дельтой. Риск записан 12.09: чередование двух моделей на этой машине само
портит замер (разброс 18-29% на декоде), поэтому разброс печатается рядом
с медианой (закон 24), а отношение считается ПОРАУНДОВО.

ФАЗА A -- рабочая конфигурация (FAST_LN по решению владельца 12.09: у
compression True пресетом, у reduction False). ФАЗА B -- норма одинаковая
в обоих плечах (оба off), чтобы отношение не смешивало цену декванта
asym sb6 с ценой рукописной нормы. ФАЗА A2 -- повтор A последним, контроль
теплового дрейфа (закон 25).

ФЛАГ ПЕРЕКЛЮЧАЕТСЯ И НА МОДЕЛИ, И НА БЛОКАХ, с del m._step_compiled --
иначе плечо off молча окажется плечом fast (ловушка 11.09). Что переключение
СОСТОЯЛОСЬ, проверяется отпечатком логитов: у fast и off он обязан
различаться, у reduction между фазами -- совпадать.
"""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

KV = "/Users/s/Develop/WKV-kvant/"
COMP = KV + os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")
RED = KV + os.environ.get("RWKVQ_RED", "reduction_1p5b_0709.rwkvq")
T = int(os.environ.get("RWKVQ_T", "512"))
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "7"))
WARM = 3


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def load(path):
    m = qm.QuantRWKV7(load_raw(path))
    print("файл %-26s пресет %-12s fast_ln %s" % (
        os.path.basename(path), str(m.preset), str(m.fast_ln)), flush=True)
    return m


def set_fast(m, v):
    m.fast_ln = v
    for b in m.blocks:
        b.fast_ln = v
    if hasattr(m, "_step_compiled"):
        del m._step_compiled


def make_pf(m, idx):
    step = m.step

    def pf():
        st = m.init_state(1)
        lg, _ = step(idx, st, True)
        return lg
    return pf


def timed(fn):
    t0 = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3


def phase(name, arms):
    fns = [(lbl, make_pf(m, IDX)) for lbl, m in arms]
    for lbl, fn in fns:
        for _ in range(WARM):
            mx.eval(fn())
    mx.synchronize()
    print("--- ФАЗА %s, T=%d, раундов %d ---" % (name, T, ROUNDS), flush=True)
    for lbl, fn in fns:
        fp = float(mx.sum(fn().astype(mx.float32)).item())
        print("  отпечаток логитов %-26s %.6f" % (lbl, fp), flush=True)
    res = dict((lbl, []) for lbl, _ in fns)
    for _ in range(ROUNDS):
        for lbl, fn in fns:
            res[lbl].append(timed(fn))
    med = {}
    for lbl, _ in fns:
        v = res[lbl]
        med[lbl] = float(np.median(v))
        print("  %-26s медиана %7.1f мс = %6.1f т/с   разброс %7.1f-%7.1f = %4.1f%%" % (
            lbl, med[lbl], 1000.0 * T / med[lbl], min(v), max(v),
            100.0 * (max(v) - min(v)) / med[lbl]), flush=True)
    a, b = fns[0][0], fns[1][0]
    rr = [res[a][i] / res[b][i] for i in range(ROUNDS)]
    print("  отношение %s / %s: по медианам %.3fx, медиана пораундовых %.3fx, разброс %.3f-%.3f" % (
        a, b, med[a] / med[b], float(np.median(rr)), min(rr), max(rr)), flush=True)
    return med


sw0 = swap_used()
print("своп до: %.1f МБ" % sw0, flush=True)
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
mc = load(COMP)
mr = load(RED)
sw1 = swap_used()
print("своп после загрузки двух файлов: %.1f МБ (дельта %+.1f)" % (sw1, sw1 - sw0),
      flush=True)

phase("A (рабочая: compression fast, reduction off)",
      [("compression fast", mc), ("reduction off", mr)])
set_fast(mc, False)
print("compression переведён в fast_ln=%s на модели и %d блоках" % (
    str(mc.fast_ln), len(mc.blocks)), flush=True)
phase("B (норма одинаковая, оба off)",
      [("compression off", mc), ("reduction off", mr)])
set_fast(mc, True)
phase("A2 (повтор A, контроль дрейфа)",
      [("compression fast", mc), ("reduction off", mr)])
sw2 = swap_used()
print("своп после: %.1f МБ (дельта за прогон %+.1f)" % (sw2, sw2 - sw0), flush=True)
print("ГОТОВО", flush=True)

