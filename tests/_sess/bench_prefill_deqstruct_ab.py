"""ГДЕ ЛЕЖАТ 131.6 МС: материализации, структура склейки или счёт?

Установлено 15.09: v3 оставляет 131.6 мс сверх пола 611.1, при трафиковом
поле декванта 32.7 мс. Питон в разности не участвует (обе фазы -- один и
тот же скомпилированный путь, трассировка выброшена), цена запусков по
записанным 11.8 мкс даёт 5-7 мс. Проба мутацией показала, что операции,
СЛИВАЮЩИЕСЯ в существующий кернель, бесплатны -- значит платить может
только за МАТЕРИАЛИЗАЦИИ, и нужна их цена ВНУТРИ графа. Прежняя «единица»
(82 мс) мерила отдельный вызов со своим eval снаружи графа и снята как
негодная.

ПЛЕЧИ НА КЕШИРОВАННЫХ ВЕСАХ, ВСЕ ТОЖДЕСТВЕННЫ ЧИСЛЕННО, поэтому отпечаток
логитов обязан совпасть у ВСЕХ фаз, включая v1 и v3 (закон 34 в сильной
форме: заглушка не только той же формы и типа, но и тех же значений):
  cache    -- деквант снят целиком, пол
  cache1   -- maximum(W, -65504): тождество для fp16-весов, но добавляет
              РОВНО ОДНУ полноразмерную материализацию
  cachecat -- половины W, тождество на каждой, и склейка: структура v3
              БЕЗ распаковочной арифметики

ПРЕДСКАЗАНИЯ, ЗАРЕГИСТРИРОВАНЫ ДО ЗАМЕРА (весов 1.208e9, w = 2.42 ГБ,
полоса 107.9 ГБ/с):
  cache1 - cache: если материализация в графе упирается в трафик, это
    2.42 записи + 2.42 чтения = +45 мс. Если выйдет 5-10 мс, трафиковая
    модель В ГРАФЕ НЕ ДЕЙСТВУЕТ, и тогда 131.6 мс -- счёт или доступ, а
    однопроходный кернель даст мало.
  cachecat - cache1: +45 мс, если concatenate есть полный барьер (лишняя
    запись и чтение), около нуля, если MLX пишет срезы склейки прямо из
    слитых производителей.
  v3 - cachecat: остаток, приписываемый распаковочной арифметике и
    чтению qblk.

Фазы, а не пораундовое чередование: флаг вморожен в mx.compile, поэтому
на плечо нужен del _step_compiled и свежий прогрев. v1 повторяется
последним как контроль дрейфа (закон 25).

V1, _SH, _sm и V3 взяты ТЕКСТУАЛЬНО из bench_prefill_deqvar_ab.py
(строки 46-59 и 82-100), а не переписаны: иначе две редакции одного
варианта разошлись бы молча (закон 30). md5 извлечённого блока печатает
сборочная команда.
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
FP16_MIN = -65504.0

V1 = gw.GwQuantLinear._dequant_w_ref  # цепочка; с 17.09 _dequant_w -- кернель
_SH = mx.array(np.array([0] * 16 + [4] * 16, dtype=np.uint8)).reshape(1, 1, 32)


def _sm(self):
    """Масштаб и минимум: мелкие (1/16 и 1/128 полного размера), берутся
    штатными view без изменений -- оптимизировать тут нечего, а риск
    расхождения есть (закон 39: dtype в измеряющем коде)."""
    OUT = self.out_features
    s = (self.qs.astype(mx.float32).reshape(OUT, self.NSB, 8)
         * self.d.astype(mx.float32)[..., None]).astype(mx.float16)
    m = (self.qm.astype(mx.float32).reshape(OUT, self.NSB, 8)
         * self.dm.astype(mx.float32)[..., None]).astype(mx.float16)
    return s.reshape(OUT, self.NB, 1), m.reshape(OUT, self.NB, 1)
def V3(self):
    """Без view: срезы qblk читаются на месте, удваиваются КОДЫ (uint8)."""
    OUT, IN, NB, xb = (self.out_features, self.in_features, self.NB,
                       self.xbits)
    if not getattr(self, "_k3", False):
        return V1(self)
    blk = self.qblk.reshape(OUT, NB, 16 + 4 * xb)
    cb2 = mx.concatenate([blk[:, :, :16], blk[:, :, :16]], axis=2)
    q = ((cb2 >> _SH) & 0xF).astype(mx.float16)
    if xb >= 1:
        b = (blk[:, :, 16:20][..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        q = q + b.reshape(OUT, NB, 32).astype(mx.float16) * 16.0
    if xb >= 2:
        b2 = (blk[:, :, 20:24][..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        q = q + b2.reshape(OUT, NB, 32).astype(mx.float16) * 32.0
    s, m = _sm(self)
    return (q * s + m).reshape(OUT, IN)



CUR = ["v1"]
_CACHE = {}


def _cached(self):
    w = _CACHE.get(id(self))
    if w is None:
        w = V1(self)
        mx.eval(w)
        _CACHE[id(self)] = w
    return w


def _patched(self):
    c = CUR[0]
    if c == "v1":
        return V1(self)
    if c == "v3":
        return V3(self)
    w = _cached(self)
    if c == "cache":
        return w
    if c == "cache1":
        return mx.maximum(w, FP16_MIN)
    if c == "cachecat":
        h = w.shape[1] // 2
        return mx.concatenate([mx.maximum(w[:, :h], FP16_MIN),
                               mx.maximum(w[:, h:], FP16_MIN)], axis=1)
    raise AssertionError(c)


gw.GwQuantLinear._dequant_w = _patched


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def make_pf(m, idx):
    step = m.step

    def pf():
        st = m.init_state(1)
        lg, _ = step(idx, st, True)
        return lg
    return pf


def phase(name, variant, m, idx):
    CUR[0] = variant
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    fn = make_pf(m, idx)
    for _ in range(WARM):
        mx.eval(fn())
    mx.synchronize()
    fp = float(mx.sum(fn().astype(mx.float32)).item())
    v = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        v.append((time.perf_counter() - t0) * 1e3)
    med = float(np.median(v))
    print("  %-26s медиана %7.1f мс = %6.1f т/с  разброс %7.1f-%7.1f = %4.1f%%  отпечаток %.6f" % (
        name, med, 1000.0 * T / med, min(v), max(v),
        100.0 * (max(v) - min(v)) / med, fp), flush=True)
    return med, fp


sw0 = swap_used()
print("своп до: %.1f МБ; T=%d, раундов %d на фазу" % (sw0, T, ROUNDS), flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
print("--- СКВОЗНОЙ ПРЕФИЛЛ, фазы ---", flush=True)
r = []
for nm, var in (("v1 (как есть)", "v1"), ("v3 (лучший вариант)", "v3"),
                ("cache (пол)", "cache"),
                ("cache1 (+1 материализация)", "cache1"),
                ("cachecat (половины+склейка)", "cachecat"),
                ("v1 повтор (дрейф)", "v1")):
    r.append((nm,) + phase(nm, var, m, IDX))
d = dict((x[0], x[1]) for x in r)
fps = set(round(x[2], 4) for x in r)
print("  отпечатки: %s" % ("СОВПАЛИ во всех фазах" if len(fps) == 1 else
                            "РАСХОДЯТСЯ, ЗАМЕР НЕДЕЙСТВИТЕЛЕН %s" % fps), flush=True)
v1, v3 = d["v1 (как есть)"], d["v3 (лучший вариант)"]
fl, c1 = d["cache (пол)"], d["cache1 (+1 материализация)"]
cc = d["cachecat (половины+склейка)"]
print("  статья при v1 %.1f мс, при v3 %.1f мс; дрейф v1 %+.1f мс" % (
    v1 - fl, v3 - fl, d["v1 повтор (дрейф)"] - v1), flush=True)
print("  ЦЕНА ОДНОЙ МАТЕРИАЛИЗАЦИИ В ГРАФЕ: %.1f мс (предсказано +45 при трафиковой модели)" % (
    c1 - fl), flush=True)
print("  ЦЕНА СКЛЕЙКИ СВЕРХ НЕЁ: %.1f мс; структура целиком %.1f мс" % (cc - c1, cc - fl), flush=True)
print("  ОСТАТОК НА АРИФМЕТИКУ И ЧТЕНИЕ qblk: %.1f мс из %.1f" % (v3 - cc, v3 - fl), flush=True)
sw1 = swap_used()
print("своп за прогон: %+.1f МБ" % (sw1 - sw0), flush=True)
print("ГОТОВО", flush=True)
