"""РАЗЛОЖЕНИЕ СТАТЬИ ДЕКВАНТА (13.09), продолжение bench_prefill_dequant_ab.

Сквозной замер дал статью 166.3 мс на префилле T=512. Здесь она
разбирается на составляющие микрозамером по ВСЕМУ стеку из 144 реальных
тензоров, и сверяется со сквозным числом.

НАПРАВЛЕНИЕ ВРАНЬЯ ИНСТРУМЕНТА (закон 29): микрозамер по всему стеку
ЗАНИЖАЕТ цену запусков, потому что 144 независимые порции работы GPU
перекрывает, а настоящий префилл есть цепочка зависимостей. Поэтому
сумма обязана сойтись со сквозной статьёй СНИЗУ, и расхождение больше
10 процентов означает, что микрозамер не описывает настоящий путь.

ТРИ ПЛЕЧА, чередуются по раундам (законы 1, 24, 25):
  deq  -- _dequant_w по всем 144 тензорам, eval на каждый (как в префилле
          каждый тензор транзиентен)
  mm   -- mx.matmul на тех же формах с УЖЕ деквантованными весами, N=512
  sq   -- квадратный matmul 4096 fp16, потолок машины на этой операции

Плюс опорное чтение mx.sum по буферу 2.4 ГБ -- полоса ЗДЕСЬ И СЕЙЧАС,
чтобы переводить миллисекунды в гигабайты, а не брать 104 из закона 12
на веру."""
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
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "5"))
WARM = 2


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def collect(root):
    """Все GwQuantLinear модели. Fused-обёртки сюда НЕ попадают и не
    должны: на префилле они не участвуют (проверено чтением 13.09)."""
    seen, out, stack = set(), [], [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, gw.GwQuantLinear):
            out.append(o)
            continue
        if isinstance(o, (list, tuple)):
            stack.extend(o)
            continue
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            stack.extend(d.values())
    return out


def qbytes(l):
    if getattr(l, "_k3", False):
        return int(l.qblk.nbytes + l.qsqm.nbytes + l.ddm.nbytes)
    n = int(l.codes.nbytes + l.qs.nbytes + l.qm.nbytes + l.d.nbytes + l.dm.nbytes)
    if l.xbits >= 1:
        n += int(l.qh.nbytes)
    if l.xbits >= 2:
        n += int(l.qh2.nbytes)
    return n


sw0 = swap_used()
print("своп до: %.1f МБ" % sw0, flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
lins = collect(m)
QB = sum(qbytes(l) for l in lins)
WB = sum(l.out_features * l.in_features * 2 for l in lins)
FLOP = sum(2 * T * l.out_features * l.in_features for l in lins)
print("тензоров %d, квантованных байт %.1f МБ, fp16 байт %.1f МБ, ФЛОП при N=%d %.3f Т" % (
    len(lins), QB / 1e6, WB / 1e6, T, FLOP / 1e12), flush=True)

BUF = mx.zeros((int(WB // 2),), dtype=mx.float16)
mx.eval(BUF)


def read_bw():
    t0 = time.perf_counter()
    mx.eval(mx.sum(BUF))
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3


W = {}
for l in lins:
    w = gw.GwQuantLinear._dequant_w(l)
    mx.eval(w)
    W[id(l)] = w
X = {}
for l in lins:
    if l.in_features not in X:
        X[l.in_features] = mx.zeros((T, l.in_features), dtype=mx.float16)
mx.eval(list(X.values()))
SA = mx.zeros((4096, 4096), dtype=mx.float16)
SB = mx.zeros((4096, 4096), dtype=mx.float16)
mx.eval(SA, SB)
sw1 = swap_used()
print("веса fp16 разложены, своп %+.1f" % (sw1 - sw0), flush=True)


def arm_deq():
    for l in lins:
        mx.eval(gw.GwQuantLinear._dequant_w(l))
    mx.synchronize()


def arm_mm():
    for l in lins:
        mx.eval(mx.matmul(X[l.in_features], W[id(l)].T))
    mx.synchronize()


def arm_sq():
    mx.eval(mx.matmul(SA, SB))
    mx.synchronize()


ARMS = [("deq (деквант 144 тензоров)", arm_deq),
        ("mm (matmul те же формы)", arm_mm),
        ("sq (matmul 4096 квадрат)", arm_sq)]


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


for _ in range(WARM):
    for _, fn in ARMS:
        fn()
    read_bw()

res = dict((lbl, []) for lbl, _ in ARMS)
bws = []
for _ in range(ROUNDS):
    for lbl, fn in ARMS:
        res[lbl].append(timed(fn))
    bws.append(read_bw())

print("--- N=%d, раундов %d, плечи чередуются ---" % (T, ROUNDS), flush=True)
med = {}
for lbl, _ in ARMS:
    v = res[lbl]
    med[lbl] = float(np.median(v))
    print("  %-28s медиана %7.2f мс   разброс %7.2f-%7.2f = %4.1f%%" % (
        lbl, med[lbl], min(v), max(v),
        100.0 * (max(v) - min(v)) / med[lbl]), flush=True)

bw = WB / (float(np.median(bws)) / 1e3) / 1e9
print("  опорное чтение %.1f МБ: %.2f мс = %.1f ГБ/с (закон 12 ждёт около 104)" % (
    WB / 1e6, float(np.median(bws)), bw), flush=True)

d = med[ARMS[0][0]]
mmv = med[ARMS[1][0]]
sq = med[ARMS[2][0]]
floor = (QB + WB) / (bw * 1e9) * 1e3
print("  ПОЛ ТРАФИКА ДЕКВАНТА: (%.1f чтение + %.1f запись) МБ / %.1f ГБ/с = %.1f мс" % (
    QB / 1e6, WB / 1e6, bw, floor), flush=True)
print("  ДЕКВАНТ: %.1f мс = %.2fx пола, подразумеваемый трафик %.1f МБ = %.2f полноразмерных проходов fp16" % (
    d, d / floor, d * bw * 1e9 / 1e3 / 1e6, (d * bw * 1e9 / 1e3 - QB) / WB), flush=True)
print("  MATMUL реальные формы: %.1f мс = %.2f ТФЛОП/с" % (
    mmv, FLOP / (mmv / 1e3) / 1e12), flush=True)
print("  MATMUL квадрат 4096:   %.2f мс = %.2f ТФЛОП/с -- потолок машины на fp16 GEMM" % (
    sq, 2.0 * 4096 ** 3 / (sq / 1e3) / 1e12), flush=True)
print("  реальные формы берут %.0f%% потолка" % (
    100.0 * (FLOP / (mmv / 1e3)) / (2.0 * 4096 ** 3 / (sq / 1e3)),), flush=True)
print("  СВЕРКА СО СКВОЗНЫМ: статья 166.3 мс (bench_prefill_dequant_ab), микрозамер %.1f мс, %.0f%%" % (
    d, 100.0 * d / 166.3), flush=True)
print("своп после: %.1f МБ (дельта %+.1f)" % (swap_used(), swap_used() - sw0), flush=True)
print("ГОТОВО", flush=True)
