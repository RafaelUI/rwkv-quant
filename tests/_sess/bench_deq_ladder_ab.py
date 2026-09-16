"""СКОЛЬКО МАТЕРИАЛИЗАЦИЙ ДЕЛАЕТ V3: считаем вычитанием, по одной операции.

Основание. 15.09: внутри графа время тождественно трафику, одна
полноразмерная материализация стоит 45.2 мс (предсказано 45.0). 16.09:
трасса MST через CLI имён кернелей не даёт, диспатчи пренебрежимы прямым
счётом (226 энкодеров на префилл = 2.7 мс из 786). Статья v3 = 142.3 мс
= 3.15 материализации, ручной счёт по тексту выражения даёт 1.4.
Разницу ищем здесь.

ПОДОЗРЕВАЕМЫЕ в V3: (a) несмежный срез blk[:, :, :16] копируется, и он
берётся ДВАЖДЫ; (b) concatenate удвоения кодов пишет 1.21 ГБ uint8 и
читает обратно; (c) astype(float16) может быть отдельным проходом
(2.42 записи + 2.42 чтения = 45 мс сам по себе); (d) развёртка битплана
на 25 тензорах; (e) запись w 2.42 ГБ -- НЕИЗБЕЖНА.

ПЛЕЧИ. Непрерывные копии срезов готовятся ЗАРАНЕЕ, вне замера, поэтому
плечо не платит за их изготовление -- ровно это и вычитается.
    cache    деквант снят (пол)
    cache1   +1 материализация (перекалибровка, ждём +45)
    v3       как есть (ждём +142, воспроизведение)
    L3       v3, но операнды склейки -- НЕПРЕРЫВНАЯ копия кодов: снимает (a)
    L2       v3, но cb2 готов НЕПРЕРЫВНЫМ целиком: снимает (a) и (b)
    L1       один проход: ниббли раскрываются броадкастом [..., None],
             без склейки и без битплана: снимает (a), (b), (d)

ЧИСЛЕННОСТЬ. L3 и L2 меняют только ОТКУДА берутся байты, арифметика на
элемент та же -- значит их отпечатки обязаны совпасть с v3 и с cache
бит-в-бит. L1 даёт ИНОЙ ПОРЯДОК нибблов (броадкаст кладёт lo,hi,lo,hi, а
формат требует lo..lo,hi..hi), поэтому его отпечаток обязан ОТЛИЧАТЬСЯ.
Прибор проверяет и то, и другое: совпадение шести и расхождение седьмого.
L1 -- плечо ВРЕМЕНИ, не кандидат в код (закон 34: форма и тип совпадают,
значения нет), в гейт не идёт.

ПРЕДСКАЗАНИЯ, ЗАРЕГИСТРИРОВАНЫ ДО ЗАМЕРА (статья сверх пола, мс):
    L3  120-131  (снятие двух копий несмежного срезa, 0.6 ГБ туда-обратно)
    L2   98-109  (плюс снятие склейки: 1.21 записи + 1.21 чтения = 22 мс)
    L1    28-34  (чтение кодов 0.60 + запись w 2.42 = 3.03 ГБ при 107 ГБ/с)
ЕСЛИ L1 ПОПАДЁТ В 28-34, то пол достижим СРЕДСТВАМИ MLX, и однопроходный
Metal-кернель на префилле не нужен вовсе -- нужна лишь перестановка
столбцов активаций под иной порядок нибблов (x стоит 1 МБ против 1.2e9
весов). Если L1 выйдет много выше -- значит astype или само чтение
держат проход, и тогда кернель оправдан.

V1, _SH, _sm, V3 взяты ТЕКСТУАЛЬНО из bench_prefill_deqvar_ab.py
(строки 46-59 и 82-100), md5 печатает сборочная команда (закон 30).
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

V1 = gw.GwQuantLinear._dequant_w
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



_SH2 = mx.array(np.array([0, 4], dtype=np.uint8)).reshape(1, 1, 1, 2)
_U8Z = mx.array(0, dtype=mx.uint8)
PRE = {}
_CACHE = {}
CUR = ["v3"]


def prep(self):
    """Непрерывные копии срезов -- ЗАРАНЕЕ, вне замера."""
    OUT, NB, xb = self.out_features, self.NB, self.xbits
    blk = self.qblk.reshape(OUT, NB, 16 + 4 * xb)
    cbp = mx.add(blk[:, :, :16], _U8Z)
    cb2p = mx.concatenate([cbp, cbp], axis=2)
    mx.eval(cbp, cb2p)
    PRE[id(self)] = (cbp, cb2p)


def _tail(self, q):
    OUT, IN = self.out_features, self.in_features
    s, m = _sm(self)
    return (q * s + m).reshape(OUT, IN)


def _bitplanes(self, q):
    OUT, NB, xb = self.out_features, self.NB, self.xbits
    blk = self.qblk.reshape(OUT, NB, 16 + 4 * xb)
    if xb >= 1:
        b = (blk[:, :, 16:20][..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        q = q + b.reshape(OUT, NB, 32).astype(mx.float16) * 16.0
    if xb >= 2:
        b2 = (blk[:, :, 20:24][..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        q = q + b2.reshape(OUT, NB, 32).astype(mx.float16) * 32.0
    return q


def L3(self):
    """v3, но операнды склейки -- непрерывная копия кодов."""
    cbp, _ = PRE[id(self)]
    cb2 = mx.concatenate([cbp, cbp], axis=2)
    q = ((cb2 >> _SH) & 0xF).astype(mx.float16)
    return _tail(self, _bitplanes(self, q))


def L2(self):
    """v3, но cb2 готов непрерывным целиком: склейки в замере нет."""
    _, cb2p = PRE[id(self)]
    q = ((cb2p >> _SH) & 0xF).astype(mx.float16)
    return _tail(self, _bitplanes(self, q))


def L1(self):
    """Один проход: ниббли броадкастом, без склейки и без битплана.
    ПОРЯДОК НИББЛОВ ИНОЙ -- плечо времени, не кандидат в код."""
    OUT, NB = self.out_features, self.NB
    cbp, _ = PRE[id(self)]
    q = ((cbp[..., None] >> _SH2) & 0xF).astype(mx.float16).reshape(OUT, NB, 32)
    return _tail(self, q)


def _cached(self):
    w = _CACHE.get(id(self))
    if w is None:
        w = V1(self)
        mx.eval(w)
        _CACHE[id(self)] = w
    return w


VAR = {"v3": V3, "L3": L3, "L2": L2, "L1": L1}


def _patched(self):
    c = CUR[0]
    if c == "cache":
        return _cached(self)
    if c == "cache1":
        return mx.maximum(_cached(self), FP16_MIN)
    return VAR[c](self)


gw.GwQuantLinear._dequant_w = _patched


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def collect(root):
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
    print("  %-22s медиана %7.1f мс  разброс %4.1f%%  отпечаток %.6f" % (
        name, med, 100.0 * (max(v) - min(v)) / med, fp), flush=True)
    return med, fp


sw0 = swap_used()
print("своп до: %.1f МБ; T=%d, раундов %d на фазу" % (sw0, T, ROUNDS), flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
lins = collect(m)
k3 = [l for l in lins if getattr(l, "_k3", False)]
print("тензоров %d, из них k3 %d" % (len(lins), len(k3)), flush=True)
assert len(k3) == len(lins), "не все тензоры k3 -- лестница написана под интерлив"
for l in lins:
    prep(l)
nb = sum(int(a.nbytes) + int(b.nbytes) for a, b in PRE.values())
print("непрерывные копии готовы вне замера: %.1f МБ, своп %+.1f" % (
    nb / 1e6, swap_used() - sw0), flush=True)
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))

print("--- СКВОЗНОЙ ПРЕФИЛЛ, фазы ---", flush=True)
r = {}
order = (("cache (пол)", "cache"), ("cache1 (+1 матер.)", "cache1"),
         ("v3 (как есть)", "v3"), ("L3 (без копий срезa)", "L3"),
         ("L2 (без склейки)", "L2"), ("L1 (один проход)", "L1"),
         ("v3 повтор (дрейф)", "v3r"))
for nm, var in order:
    r[nm] = phase(nm, "v3" if var == "v3r" else var, m, IDX)

fl = r["cache (пол)"][0]
exact = [nm for nm, _ in order if nm != "L1 (один проход)"]
fps = set(round(r[nm][1], 4) for nm in exact)
print("  отпечатки шести точных плеч: %s" % (
    "СОВПАЛИ" if len(fps) == 1 else "РАСХОДЯТСЯ, НЕДЕЙСТВИТЕЛЬНО %s" % fps), flush=True)
print("  отпечаток L1 отличается: %s (обязан, порядок нибблов иной)" % (
    "да" if round(r["L1 (один проход)"][1], 4) not in fps else "НЕТ, ЧТО-ТО НЕ ТАК"), flush=True)
print("--- СТАТЬЯ СВЕРХ ПОЛА, мс ---", flush=True)
for nm, _ in order:
    art = r[nm][0] - fl
    print("  %-22s %7.1f  = %.2f материализации" % (nm, art, art / 45.2), flush=True)
v3a = r["v3 (как есть)"][0] - fl
print("  СНИМАЕТ: копии срезa %.1f мс, склейка %.1f мс, битплан+порядок %.1f мс" % (
    v3a - (r["L3 (без копий срезa)"][0] - fl),
    (r["L3 (без копий срезa)"][0] - r["L2 (без склейки)"][0]),
    (r["L2 (без склейки)"][0] - r["L1 (один проход)"][0])), flush=True)
print("  дрейф v3 %+.1f мс; своп за прогон %+.1f МБ" % (
    r["v3 повтор (дрейф)"][0] - r["v3 (как есть)"][0], swap_used() - sw0), flush=True)
print("ГОТОВО", flush=True)
