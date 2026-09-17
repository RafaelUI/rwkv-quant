"""ПРОВЕРКА ТЕОРИИ 13.09: статья декванта 166 мс = 5.1x пола трафика,
и разрыв сидит НЕ в GEMM, а в форме самой цепочки _dequant_w.

ДИАГНОЗ, поставленный чтением:
  (1) self.codes / self.qh / self.qh2 в режиме K3 -- это ленивые view,
      каждый делает reshape НЕПРЕРЫВНОГО срезa интерлив-буфера qblk
      (stride 16+4*xbits), а reshape несмежного срезa обязан копировать.
      Три полноразмерные копии до начала арифметики.
  (2) mx.concatenate нибблов стоит барьером слияния в самом начале
      цепочки, поэтому полный размер материализуется ещё раз, ДО того
      как к нему применят битпланы и масштаб.

ДВА КАНДИДАТА:
  v2 -- concatenate переезжает в КОНЕЦ: каждая половина нибблов считается
        до готового w, склейка одна и последняя.
  v3 -- view не используются вовсе, срезы qblk читаются на месте без
        reshape, а вместо склейки половин удваиваются КОДЫ (uint8,
        вдвое дешевле fp16) и дальше вся цепочка однородна и слипаема.

ПРЕДСКАЗАНИЕ, ЗАРЕГИСТРИРОВАННОЕ ДО ЗАМЕРА: v3 дешевле v2, потому что
несёт 4.0 ГБ записи против 5.4, и оба дешевле v1 (6 проходов). Пол
остаётся 32.7 мс, ни один вариант его не достигает -- для этого нужен
один кернель.

БИТ-В-БИТ ОБЯЗАТЕЛЬНО: поэлементно все три варианта делают те же
операции в том же типе над теми же числами, меняется только порядок
склейки, поэтому порог здесь не точность, а равенство разрядов
(закон 37). Гейт сверяет uint16-представление на ВСЕХ тензорах модели, а
не на мелких (закон 31), и печатает, сколько тензоров каждой
разновидности xbits проверено.

Арбитр -- сквозной префилл (закон 29), не микрозамер."""
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


def V2(self):
    """concatenate в конце: половины нибблов доводятся до готового w."""
    OUT, IN, NB = self.out_features, self.in_features, self.NB
    cb = self.codes.reshape(OUT, NB, 16)
    s, m = _sm(self)
    halves = []
    for half in (0, 1):
        q = (cb & 0xF).astype(mx.float16) if half == 0 else (cb >> 4).astype(mx.float16)
        if self.xbits >= 1:
            h = self.qh.reshape(OUT, NB, 4)[:, :, 2 * half:2 * half + 2]
            b = (h[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
            q = q + b.reshape(OUT, NB, 16).astype(mx.float16) * 16.0
        if self.xbits >= 2:
            h2 = self.qh2.reshape(OUT, NB, 4)[:, :, 2 * half:2 * half + 2]
            b2 = (h2[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
            q = q + b2.reshape(OUT, NB, 16).astype(mx.float16) * 32.0
        halves.append(q * s + m)
    return mx.concatenate(halves, axis=2).reshape(OUT, IN)


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


VAR = {"v1": V1, "v2": V2, "v3": V3}
CUR = ["v1"]
_CACHE = {}


def _patched(self):
    if CUR[0] == "cache":
        w = _CACHE.get(id(self))
        if w is None:
            w = V1(self)
            mx.eval(w)
            _CACHE[id(self)] = w
        return w
    return VAR[CUR[0]](self)


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
    print("  %-24s медиана %7.1f мс = %6.1f т/с  разброс %7.1f-%7.1f = %4.1f%%  отпечаток %.6f" % (
        name, med, 1000.0 * T / med, min(v), max(v),
        100.0 * (max(v) - min(v)) / med, fp), flush=True)
    return med, fp


sw0 = swap_used()
print("своп до: %.1f МБ" % sw0, flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
lins = collect(m)
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))

print("--- ГЕЙТ БИТ-В-БИТ по всем %d тензорам ---" % len(lins), flush=True)
cnt = {}
bad = []
for l in lins:
    a = mx.view(V1(l), mx.uint16)
    for nm, f in (("v2", V2), ("v3", V3)):
        b = mx.view(f(l), mx.uint16)
        ok = bool(mx.all(a == b).item())
        if not ok:
            bad.append((nm, l.out_features, l.in_features, l.xbits))
    k = (l.xbits, bool(getattr(l, "_k3", False)))
    cnt[k] = cnt.get(k, 0) + 1
    mx.eval(a)
    del a
for k in sorted(cnt):
    print("  xbits=%d k3=%s: %d тензоров" % (k[0], k[1], cnt[k]), flush=True)
print("  расхождений: %d %s" % (len(bad), bad[:4] if bad else ""), flush=True)
assert not bad, "варианты НЕ бит-в-бит, замер смысла не имеет"

print("--- СКВОЗНОЙ ПРЕФИЛЛ T=%d, %d раундов на фазу ---" % (T, ROUNDS), flush=True)
r = []
for nm, var in (("v1 (как есть)", "v1"), ("v2 (склейка в конце)", "v2"),
                ("v3 (без view)", "v3"), ("cache (пол: деквант снят)", "cache"),
                ("v1 повтор (дрейф)", "v1")):
    r.append((nm,) + phase(nm, var, m, IDX))

fps = set(x[2] for x in r)
print("  отпечатки: %s" % ("СОВПАЛИ все пять фаз" if len(fps) == 1
                            else "РАСХОДЯТСЯ, ЗАМЕР НЕДЕЙСТВИТЕЛЕН %s" % fps), flush=True)
base, floor = r[0][1], r[3][1]
art = base - floor
print("  статья декванта при v1: %.1f мс; дрейф v1 -> v1 повтор %+.1f мс" % (
    art, r[4][1] - base), flush=True)
for nm, med, _ in r[1:3]:
    print("  %-24s снимает %6.1f мс = %4.0f%% статьи, остаток статьи %6.1f мс" % (
        nm, base - med, 100.0 * (base - med) / art, med - floor), flush=True)
print("своп после: %.1f МБ (дельта %+.1f)" % (swap_used(), swap_used() - sw0), flush=True)
print("ГОТОВО", flush=True)
