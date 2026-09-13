"""ЦЕЛЬ ЧИСЛОМ для приоритета 1: сколько из префилла compression стоит
САМ деквант, а сколько -- matmul и остальное.

На префилле (N >= GEMM_MIN_BATCH_NB=128) GwQuantLinear.__call__ идёт
плотным путём: _dequant_w() материализует ВЕСЬ тензор в fp16 (плюс
полноразмерные временные внутри), затем mx.matmul. Тайловый GEMM с
декодом в threadgroup-памяти убирает ровно первую половину.

Плечи чередуются в ОДНОМ процессе по раундам (законы 1 и 25), разброс
печатается рядом с медианой (закон 24), своп мерится дельтой (закон 11).

ЗАГЛУШКА ПОВТОРЯЕТ dtype И ФОРМУ (закон 34): плечо "кеш" возвращает РОВНО
тот тензор, который посчитал бы настоящий _dequant_w, только один раз.
Поэтому отпечатки логитов двух плеч обязаны совпасть БИТ-В-БИТ -- это и
есть проверка, что заглушка не подменила математику. Расхождение
отпечатков делает замер недействительным.

ЧЕГО ЭТОТ ЗАМЕР НЕ ГОВОРИТ: плечо "кеш" не есть потолок тайлового ядра.
Оно читает веса в fp16 (2 байта на вес), а тайловое ядро будет читать
квантованные (около 0.625). Плечо "кеш" -- это пол ТЕКУЩЕЙ конструкции,
то есть нижняя граница цели, а не верхняя.
"""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.formats.reader import load_raw

KV = "/Users/s/Develop/WKV-kvant/"
COMP = KV + os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")
T = int(os.environ.get("RWKVQ_T", "512"))
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "7"))
WARM = 3

_REAL = gw.GwQuantLinear._dequant_w
_CACHE = {}
USE_CACHE = [False]


def _patched(self):
    if not USE_CACHE[0]:
        return _REAL(self)
    w = _CACHE.get(id(self))
    if w is None:
        w = _REAL(self)
        mx.eval(w)
        _CACHE[id(self)] = w
    return w


gw.GwQuantLinear._dequant_w = _patched


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def load(path):
    m = qm.QuantRWKV7(load_raw(path))
    print("файл %-24s пресет %-12s fast_ln %s" % (
        os.path.basename(path), str(m.preset), str(m.fast_ln)), flush=True)
    return m


def make_pf(m, idx):
    step = m.step

    def pf():
        st = m.init_state(1)
        lg, _ = step(idx, st, True)
        return lg
    return pf


def timed(fn, flag):
    USE_CACHE[0] = flag
    t0 = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3


def fp_of(fn, flag):
    USE_CACHE[0] = flag
    return float(mx.sum(fn().astype(mx.float32)).item())


def fresh(flag):
    """Свежая компиляция под текущий флаг: mx.compile вмораживает ветку
    в трассировку, поэтому смена флага без del _step_compiled НЕ доходит
    (проверено нулевым прогоном 13.09: кеш остался пуст, разность -0.3 мс)."""
    USE_CACHE[0] = flag
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    return make_pf(m, IDX)


def phase(name, flag):
    fn = fresh(flag)
    for _ in range(WARM):
        mx.eval(fn())
    mx.synchronize()
    fp = fp_of(fn, flag)
    v = [timed(fn, flag) for _ in range(ROUNDS)]
    med = float(np.median(v))
    print("  %-28s медиана %7.1f мс = %6.1f т/с  разброс %7.1f-%7.1f = %4.1f%%  отпечаток %.6f" % (
        name, med, 1000.0 * T / med, min(v), max(v),
        100.0 * (max(v) - min(v)) / med, fp), flush=True)
    return med, fp


sw0 = swap_used()
print("своп до: %.1f МБ" % sw0, flush=True)
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
m = load(COMP)
print("--- T=%d, раундов %d на фазу ---" % (T, ROUNDS), flush=True)

mA, fA = phase("A: как есть", False)

USE_CACHE[0] = True
st = m.init_state(1)
lg, _ = m.forward_stateful(IDX, st, True)
mx.eval(lg)
nb = sum(int(w.nbytes) for w in _CACHE.values())
sw1 = swap_used()
print("кеш деквантованных весов собран ВНЕ компиляции: %d тензоров, %.1f МБ, своп %+.1f" % (
    len(_CACHE), nb / 1e6, sw1 - sw0), flush=True)
assert len(_CACHE) > 0, "кеш пуст -- заглушка не исполнилась, замер недействителен"

mB, fB = phase("B: деквант кеширован", True)
mA2, fA2 = phase("A2: как есть (повтор)", False)

print("  отпечатки: A %.6f  B %.6f  A2 %.6f  --  %s" % (
    fA, fB, fA2,
    "СОВПАЛИ" if (fA == fB and fA == fA2) else "РАСХОДЯТСЯ, ЗАМЕР НЕДЕЙСТВИТЕЛЕН"), flush=True)
base = min(mA, mA2)
print("  СТАТЬЯ ДЕКВАНТА: %.1f мс из %.1f (%.1f%% префилла); против A2: %.1f мс" % (
    mA - mB, mA, 100.0 * (mA - mB) / mA, mA2 - mB), flush=True)
print("  дрейф A -> A2: %.1f мс = %.1f%%" % (mA2 - mA, 100.0 * (mA2 - mA) / mA), flush=True)
sw2 = swap_used()
print("своп после: %.1f МБ (дельта за прогон %+.1f)" % (sw2, sw2 - sw0), flush=True)
print("ГОТОВО", flush=True)
