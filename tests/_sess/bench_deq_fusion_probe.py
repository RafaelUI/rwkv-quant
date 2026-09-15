"""СЛИПАЕТ ЛИ mx.compile ЦЕПОЧКУ ДЕКВАНТА? Проверка мутацией.

v3 (лучший вариант 13.09) снимает лишь 25% статьи, остаток 131.6 мс при
поле трафика 32.7. Арифметика подсказывает объяснение: для 120 тензоров
из 145 (xbits=0) цепочка это concat кодов (1.34 ГБ), затем shift/and/
astype (2.68), затем q*s+m (2.68) -- если каждая стадия материализуется
отдельно, выходит около 13.4 ГБ и 124 мс, что и наблюдается.

ПРЕДСКАЗАНИЕ, ЗАРЕГИСТРИРОВАННОЕ ДО ЗАМЕРА. Вставляем в цепочку ХОЛОСТЫЕ
поэлементные операции над полноразмерным fp16 (q+0.0, ещё раз, ещё раз).
  -- если mx.compile НЕ слипает: каждая добавка стоит чтение+запись
     2.68+2.68 ГБ = 5.37/107.9 = около 50 мс, наклон линейный;
  -- если слипает: добавки бесплатны, 0-5 мс, и тогда остаток 131.6 мс
     объясняется не проходами, а счётом, и один кернель пола не даст.
От наклона зависит, стоит ли писать кернель декванта: он окупается
ТОЛЬКО в первом случае.

q + 0.0 в fp16 бит-в-бит тождественно q для всех конечных значений и для
нуля, а значений вне решётки тут нет (коды 0..63, масштаб конечен),
поэтому отпечатки всех фаз обязаны совпасть -- это и есть проверка, что
добавка ничего не меняет кроме числа проходов."""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.formats.reader import load_raw

KV = "/Users/s/Develop/WKV-kvant/"
COMP = KV + "compression_v2_cand.rwkvq"
T, ROUNDS, WARM = 512, 7, 3
V1 = gw.GwQuantLinear._dequant_w
_SH = mx.array(np.array([0] * 16 + [4] * 16, dtype=np.uint8)).reshape(1, 1, 32)
NOOP = [0]


def V3n(self):
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
    for _ in range(NOOP[0]):
        q = mx.maximum(q, mx.array(-65504.0, dtype=mx.float16))
    s = (self.qs.astype(mx.float32).reshape(OUT, self.NSB, 8)
         * self.d.astype(mx.float32)[..., None]).astype(mx.float16)
    mm = (self.qm.astype(mx.float32).reshape(OUT, self.NSB, 8)
          * self.dm.astype(mx.float32)[..., None]).astype(mx.float16)
    return (q * s.reshape(OUT, NB, 1) + mm.reshape(OUT, NB, 1)).reshape(OUT, IN)


gw.GwQuantLinear._dequant_w = V3n


def swap_used():
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                       capture_output=True, text=True).stdout.split()
    return float(o[5].rstrip("M"))


sw0 = swap_used()
print("своп до: %.1f МБ" % sw0, flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
print("--- T=%d, %d раундов на фазу ---" % (T, ROUNDS), flush=True)
out = []
for n in (0, 1, 2, 0):
    NOOP[0] = n
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    step = m.step

    def pf():
        st = m.init_state(1)
        lg, _ = step(IDX, st, True)
        return lg
    for _ in range(WARM):
        mx.eval(pf())
    mx.synchronize()
    fp = float(mx.sum(pf().astype(mx.float32)).item())
    v = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        mx.eval(pf())
        mx.synchronize()
        v.append((time.perf_counter() - t0) * 1e3)
    med = float(np.median(v))
    out.append((n, med, fp))
    print("  холостых операций %d: медиана %7.1f мс  разброс %7.1f-%7.1f = %4.1f%%  отпечаток %.6f" % (
        n, med, min(v), max(v), 100.0 * (max(v) - min(v)) / med, fp), flush=True)
print("  отпечатки: %s" % ("СОВПАЛИ" if len(set(x[2] for x in out)) == 1
                            else "РАСХОДЯТСЯ, ЗАМЕР НЕДЕЙСТВИТЕЛЕН"), flush=True)
print("  цена одной холостой операции: %+.1f мс (0->1), %+.1f мс (1->2); дрейф %+.1f" % (
    out[1][1] - out[0][1], out[2][1] - out[1][1], out[3][1] - out[0][1]), flush=True)
print("  ожидание при отсутствии слияния: около +50 мс на операцию", flush=True)
print("  форма мутации: maximum(q, -65504) -- тождество для всех значений (коды 0..63), но НЕ сворачиваемое константным фолдингом, в отличие от q+0.0", flush=True)
print("своп после: %.1f МБ (дельта %+.1f)" % (swap_used(), swap_used() - sw0), flush=True)
print("ГОТОВО", flush=True)
