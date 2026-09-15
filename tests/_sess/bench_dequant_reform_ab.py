"""ПЕРЕПИСЬ _dequant_w В MLX: снять лишние временные, не трогая арифметику.

Замер 13.09 (bench_dequant_fusion_ab): compiled деквант стоит 2.64
полноразмерных прохода при неизбежных 0.66, а тензоры с битпланом qh --
3.66 против 2.23 без него. Лишнее: (1) codes и qh достаются из
интерлива qblk через reshape(OUT, IN//2), что заставляет копию;
(2) concatenate нибблов материализует весь тензор ДО умножения на
масштаб; (3) битплан разворачивается в uint8 [OUT, IN] целиком.

ПЕРЕПИСЬ: считать двумя половинами (низкие ниббли, высокие), брать
коды и битплан срезами qblk напрямую, применять масштаб и сдвиг ДО
склейки, склеивать один раз в конце. Порядок операций НА ЭЛЕМЕНТ тот
же, поэтому требуется БИТ-В-БИТ, и это не подгонка порога, а свойство
правки (закон 37: порог ниже решётки типа проверяет бит-в-бит -- здесь
именно это и нужно).

ПРЕДСКАЗАНИЕ, ЗАРЕГИСТРИРОВАНО ДО ЗАМЕРА: остаётся одна обязательная
материализация половин (2 Б/вес) плюс склейка (2 читает, 2 пишет), то
есть около 1.6-1.7 единицы против 2.64, на тензорах с qh выигрыш больше.
Склейку убрать без смены порядка накопления matmul нельзя -- это
отдельный вопрос и отдельное решение.

ГЕЙТ ИДЁТ ПЕРВЫМ, до замера, на ВСЕХ 144 реальных тензорах (закон 17:
формы брать из настоящего файла; закон 31: покрыты обе разновидности --
xbits 0 и xbits 1).
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
_REAL = gw.GwQuantLinear._dequant_w


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def dq_new(self):
    """Та же математика на элемент, меньше материализаций."""
    OUT, IN = self.out_features, self.in_features
    NB, NSB, xb = self.NB, self.NSB, self.xbits
    assert self._k3, "перепись написана под интерлив k3"
    blk = self.qblk.reshape(OUT, NB, 16 + 4 * xb)
    sh = mx.arange(8, dtype=mx.uint8)
    s = (self.qs.astype(mx.float32).reshape(OUT, NSB, 8)
         * self.d.astype(mx.float32)[..., None]).astype(mx.float16)
    m = (self.qm.astype(mx.float32).reshape(OUT, NSB, 8)
         * self.dm.astype(mx.float32)[..., None]).astype(mx.float16)
    s = s.reshape(OUT, NB, 1)
    m = m.reshape(OUT, NB, 1)
    cb = blk[:, :, :16]
    halves = []
    for h in (0, 1):
        q = (cb & 0xF if h == 0 else cb >> 4).astype(mx.float16)
        if xb >= 1:
            b1 = blk[:, :, 16 + 2 * h:18 + 2 * h]
            q = q + (((b1[..., None] >> sh) & 1).reshape(OUT, NB, 16)
                     .astype(mx.float16) * 16.0)
        if xb >= 2:
            b2 = blk[:, :, 20 + 2 * h:22 + 2 * h]
            q = q + (((b2[..., None] >> sh) & 1).reshape(OUT, NB, 16)
                     .astype(mx.float16) * 32.0)
        halves.append(q * s + m)
    return mx.concatenate(halves, axis=2).reshape(OUT, IN)


sw0 = swap_used()
print("своп до: %.1f МБ" % sw0, flush=True)
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
m = qm.QuantRWKV7(load_raw(COMP))
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
print("тензоров: %d, весов %.3fe9" % (n, nw / 1e9), flush=True)

bad = []
for i, l in enumerate(LINS):
    a, b = _REAL(l), dq_new(l)
    mx.eval(a, b)
    okk = (a.shape == b.shape and a.dtype == b.dtype
           and bool(mx.array_equal(a, b).item()))
    if not okk:
        bad.append((i, l.xbits, l.out_features, l.in_features))
    del a, b
byxb = {}
for l in LINS:
    byxb[l.xbits] = byxb.get(l.xbits, 0) + 1
print("ГЕЙТ БИТ-В-БИТ на %d тензорах, разновидности xbits %s: %s" % (
    n, sorted(byxb.items()), "ЗЕЛЁНЫЙ" if not bad else ("КРАСНЫЙ %s" % bad[:5])), flush=True)
assert not bad, "перепись не бит-в-бит, замер смысла не имеет"

W = []
for l in LINS:
    w = _REAL(l)
    mx.eval(w)
    W.append(w)
print("кеш для единицы: %.1f МБ, своп %+.1f" % (
    sum(int(w.nbytes) for w in W) / 1e6, swap_used() - sw0), flush=True)
C_REF = [mx.compile(lambda l=l: _REAL(l)) for l in LINS]
C_NEW = [mx.compile(lambda l=l: dq_new(l)) for l in LINS]
ARMS = [("ref compiled", lambda i: mx.eval(C_REF[i]())),
        ("перепись compiled", lambda i: mx.eval(C_NEW[i]())),
        ("единица (mx.abs)", lambda i: mx.eval(mx.abs(W[i])))]

for name, fn in ARMS:
    for i in range(n):
        fn(i)
mx.synchronize()
tot = dict((nm, []) for nm, _ in ARMS)
per = dict((nm, [[] for _ in range(n)]) for nm, _ in ARMS)
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

print("--- %d раундов, плечи чередуются ---" % ROUNDS, flush=True)
med = {}
for name, _ in ARMS:
    v = tot[name]
    med[name] = float(np.median(v))
    print("  %-20s медиана %7.1f мс, разброс %7.1f-%7.1f = %4.1f%%" % (
        name, med[name], min(v), max(v), 100.0 * (max(v) - min(v)) / med[name]), flush=True)
u = med["единица (mx.abs)"]
print("  в единицах: ref %.2f, перепись %.2f (неизбежное 0.66)" % (
    med["ref compiled"] / u, med["перепись compiled"] / u), flush=True)
print("  СНЯТО: %.1f мс = %.0f%% декванта; ожидалось 1.6-1.7 ед, вышло %.2f" % (
    med["ref compiled"] - med["перепись compiled"],
    100.0 * (1 - med["перепись compiled"] / med["ref compiled"]),
    med["перепись compiled"] / u), flush=True)
for xb in sorted(byxb):
    idx = [i for i in range(n) if LINS[i].xbits == xb]
    sr = sum(float(np.median(per["ref compiled"][i])) for i in idx)
    sn = sum(float(np.median(per["перепись compiled"][i])) for i in idx)
    su = sum(float(np.median(per["единица (mx.abs)"][i])) for i in idx)
    print("    xbits %d: ref %6.1f мс (%.2f ед) -> перепись %6.1f мс (%.2f ед)" % (
        xb, sr, sr / su, sn, sn / su), flush=True)
print("своп после: %.1f МБ (дельта %+.1f)" % (swap_used(), swap_used() - sw0), flush=True)
print("ГОТОВО", flush=True)
