"""СКВОЗНОЙ ЗАМЕР ОДНОПРОХОДНОГО КЕРНЕЛЯ ДЕКВАНТА против цепочек MLX.

Гейт пройден ДО замера: tests/test_gw_dequant_kernel_parity.py --
бит-в-бит на 145 реальных тензорах, и он КРАСНЕЕТ на мутации
(MUTATE=1), то есть разрешающая способность подтверждена (закон 37).

Цель, поставленная числом 16.09: статья 146.0 -> 30-40 мс.
Отпечатки логитов ВСЕХ фаз обязаны совпасть: кернель бит-в-бит.
"""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.backends.metal.gw_dequant_kernel import dequant_w
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



_CACHE = {}
CUR = ["v1"]


def _patched(self):
    c = CUR[0]
    if c == "v1":
        return V1(self)
    if c == "v3":
        return V3(self)
    if c == "kern":
        return dequant_w(self)
    w = _CACHE.get(id(self))
    if w is None:
        w = V1(self)
        mx.eval(w)
        _CACHE[id(self)] = w
    return w


gw.GwQuantLinear._dequant_w = _patched


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def phase(name, variant, m, idx):
    CUR[0] = variant
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    step = m.step

    def pf():
        st = m.init_state(1)
        lg, _ = step(idx, st, True)
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
    print("  %-22s медиана %7.1f мс = %6.1f т/с  разброс %4.1f%%  отпечаток %.6f" % (
        name, med, 1000.0 * T / med, 100.0 * (max(v) - min(v)) / med, fp), flush=True)
    return med, fp


sw0 = swap_used()
print("своп до: %.1f МБ; T=%d, раундов %d" % (sw0, T, ROUNDS), flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
rng = np.random.default_rng(0)
IDX = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
print("--- фазы ---", flush=True)
r = {}
order = (("v1 (как есть)", "v1"), ("v3 (лучший MLX)", "v3"),
         ("КЕРНЕЛЬ", "kern"), ("cache (пол)", "cache"),
         ("v1 повтор (дрейф)", "v1"))
for nm, var in order:
    r[nm] = phase(nm, var, m, IDX)
fps = set(round(x[1], 4) for x in r.values())
print("  отпечатки: %s" % ("СОВПАЛИ во всех фазах" if len(fps) == 1 else
                            "РАСХОДЯТСЯ, НЕДЕЙСТВИТЕЛЬНО %s" % fps), flush=True)
fl = r["cache (пол)"][0]
for nm, _ in order:
    print("  статья %-22s %7.1f мс" % (nm, r[nm][0] - fl), flush=True)
k = r["КЕРНЕЛЬ"][0]
print("  КЕРНЕЛЬ против v1: %+.1f мс (%.1f%%), против v3: %+.1f мс; цель была 30-40 мс статьи" % (
    k - r["v1 (как есть)"][0], 100.0 * (k - r["v1 (как есть)"][0]) / r["v1 (как есть)"][0],
    k - r["v3 (лучший MLX)"][0]), flush=True)
print("  дрейф %+.1f мс; своп %+.1f МБ" % (
    r["v1 повтор (дрейф)"][0] - r["v1 (как есть)"][0], swap_used() - sw0), flush=True)
print("ГОТОВО", flush=True)
