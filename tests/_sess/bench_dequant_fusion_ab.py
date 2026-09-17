"""ПОЧЕМУ статья декванта стоит 166 мс: временные тензоры или счёт?

Гипотеза (13.09, НЕ проверена на момент написания прибора): mx.compile НЕ
сливает цепочку _dequant_w, и статья -- это трафик полноразмерных
временных (concatenate кодов, битпланы qh/qh2 развёрнутые в [OUT, IN],
q*s+m). Против неё: префилл целиком идёт под mx.compile, а 12.09
записано, что цепочку пред-WKV mx.compile свёл примерно к одному
диспатчу.

ПРЕДСКАЗАНИЕ, ЗАРЕГИСТРИРОВАНО ДО ЗАМЕРА. Неизбежная часть декванта --
чтение 0.625 Б/вес плюс запись 2 Б/вес; на 1.21e9 весов это 3.2 ГБ = 30 мс
при полосе 104 ГБ/с (закон 12). Измеренная статья 166 мс, излишек 136 мс
= 14 ГБ = 9 Б/вес = 4-5 полноразмерных проходов. Отсюда:
  -- если цепочка НЕ слита: плечо compiled примерно равно eager и оба
     дают 4-6 единиц ниже определённой ЕДИНИЦЫ;
  -- если слита: compiled сильно меньше eager и лежит около 1-2 единиц,
     а тогда 166 мс объясняются НЕ временными, и гипотезу надо снять.

ЕДИНИЦА -- цена ОДНОГО полноразмерного прохода по весам в fp16: mx.abs
на уже деквантованном тензоре, то есть 2 Б/вес чтение плюс 2 Б/вес
запись. Арифметика при 104 ГБ/с: 4 * 1.21e9 / 104e9 = 46 мс. Измеряется
тем же прибором, что и плечи, поэтому отношение статья/единица
безразмерно и не зависит от того, соврала ли абсолютная шкала (закон 29).

СВЕРКА СО СКВОЗНЫМ ЗАМЕРОМ ОБЯЗАТЕЛЬНА: плечо compiled должно
воспроизвести сквозную статью 166.3-179.0 мс (bench_prefill_dequant_ab,
13.09). Если не воспроизводит -- микрозамер врёт, и читать надо только
отношения, о чём и печатается вердикт.

Закон 29: микрозамер на матрицах ОДНОГО слоя завышает (набор влезает в
кэш), поэтому каждое плечо проходит ВСЕ 144 тензора за раунд -- рабочий
набор 2.4 ГБ, кэш побеждён. Закон 4: eval на каждой итерации, иначе
ленивый граф накопит все 2.4 ГБ. Плечи чередуются по раундам (закон 1),
разброс печатается рядом с медианой (закон 24), своп дельтой (закон 11).
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
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "5"))
BW = 104.0
E2E = (166.3, 179.0)
_REAL = gw.GwQuantLinear._dequant_w_ref  # цепочка; с 17.09 _dequant_w -- кернель


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


sw0 = swap_used()
print("своп до: %.1f МБ" % sw0, flush=True)
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
m = qm.QuantRWKV7(load_raw(COMP))
print("файл %s пресет %s fast_ln %s" % (
    os.path.basename(COMP), str(m.preset), str(m.fast_ln)), flush=True)

SEEN = []


def _rec(self):
    SEEN.append(self)
    return _REAL(self)


gw.GwQuantLinear._dequant_w = _rec
st = m.init_state(1)
lg, _ = m.forward_stateful(IDX, st, True)
mx.eval(lg)
gw.GwQuantLinear._dequant_w = _REAL

LINS, ids = [], set()
for l in SEEN:
    if id(l) not in ids:
        ids.add(id(l))
        LINS.append(l)
n = len(LINS)
nw = sum(l.in_features * l.out_features for l in LINS)
print("тензоров деквантовано за проход префилла: %d вызовов, %d различных" % (
    len(SEEN), n), flush=True)
assert len(SEEN) == n, "тензор деквантуется дважды за проход -- это отдельная находка"
assert n > 100, ("собрано слишком мало тензоров", n)
print("весов всего %.3fe9; единица (4 Б/вес при %.0f ГБ/с) = %.1f мс по арифметике" % (
    nw / 1e9, BW, 4.0 * nw / BW / 1e6), flush=True)

W = []
for l in LINS:
    w = _REAL(l)
    mx.eval(w)
    W.append(w)
assert W[0].dtype == mx.float16, ("единица мерила бы не тот тип", W[0].dtype)
assert mx.abs(W[0]).dtype == mx.float16, "mx.abs сменил тип -- единица недействительна"
nb = sum(int(w.nbytes) for w in W)
sw1 = swap_used()
print("кеш деквантованных весов: %.1f МБ, своп %+.1f" % (nb / 1e6, sw1 - sw0), flush=True)

COMPILED = [mx.compile(lambda l=l: _REAL(l)) for l in LINS]


def a_eager(i):
    mx.eval(_REAL(LINS[i]))


def a_comp(i):
    mx.eval(COMPILED[i]())


def a_unit(i):
    mx.eval(mx.abs(W[i]))


ARMS = [("eager", a_eager), ("compiled", a_comp), ("единица (mx.abs)", a_unit)]


for name, fn in ARMS:
    for i in range(n):
        fn(i)
mx.synchronize()

tot = dict((name, []) for name, _ in ARMS)
per = dict((name, [[] for _ in range(n)]) for name, _ in ARMS)
for r in range(ROUNDS):
    for name, fn in ARMS:
        s = 0.0
        for i in range(n):
            t0 = time.perf_counter()
            fn(i)
            dt = (time.perf_counter() - t0) * 1e3
            per[name][i].append(dt)
            s += dt
        tot[name].append(s)
mx.synchronize()

print("--- %d тензоров, %d раундов, плечи чередуются по раундам ---" % (n, ROUNDS), flush=True)
med = {}
for name, _ in ARMS:
    v = tot[name]
    med[name] = float(np.median(v))
    print("  %-18s сумма по всем тензорам: медиана %7.1f мс, разброс %7.1f-%7.1f = %4.1f%%" % (
        name, med[name], min(v), max(v),
        100.0 * (max(v) - min(v)) / med[name]), flush=True)

u = med["единица (mx.abs)"]
print("  В ЕДИНИЦАХ ПОЛНОРАЗМЕРНОГО ПРОХОДА: eager %.2f, compiled %.2f" % (
    med["eager"] / u, med["compiled"] / u), flush=True)
print("  compiled / eager = %.3f" % (med["compiled"] / med["eager"]), flush=True)
print("  СВЕРКА СО СКВОЗНЫМ: compiled %.1f мс против сквозной статьи %.1f-%.1f мс" % (
    med["compiled"], E2E[0], E2E[1]), flush=True)
ok = E2E[0] * 0.7 <= med["compiled"] <= E2E[1] * 1.3
print("  вердикт сверки: %s" % (
    "воспроизводит, абсолюты читать можно" if ok else
    "НЕ воспроизводит, читать ТОЛЬКО отношения"), flush=True)

print("  разбивка по xbits (4/5/6 бит = xbits 0/1/2), медианы по тензору:", flush=True)
for xb in (0, 1, 2):
    idx = [i for i in range(n) if LINS[i].xbits == xb]
    if not idx:
        continue
    gw_ = sum(LINS[i].in_features * LINS[i].out_features for i in idx)
    se = sum(float(np.median(per["eager"][i])) for i in idx)
    sc = sum(float(np.median(per["compiled"][i])) for i in idx)
    su = sum(float(np.median(per["единица (mx.abs)"][i])) for i in idx)
    print("    xbits %d: %3d тензоров, %.3fe9 весов -- eager %6.1f мс (%.2f ед), compiled %6.1f мс (%.2f ед)" % (
        xb, len(idx), gw_ / 1e9, se, se / su, sc, sc / su), flush=True)
sw2 = swap_used()
print("своп после: %.1f МБ (дельта за прогон %+.1f)" % (sw2, sw2 - sw0), flush=True)
print("ГОТОВО", flush=True)
