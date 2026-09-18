"""РАЗЛОЖЕНИЕ ШАГА ДЕКОДА В ОДНОМ ПРОЦЕССЕ (18.09, вечер) на нынешнем рабочем пути.

Зачем: бюджет 10.09 снят (1) до перехода норм на mx.fast.layer_norm 12.09,
(2) частично прокси-цепочками (нижние границы), (3) до того, как 17.09
поймана ловушка ленивого MLX. Здесь -- аблация на НАСТОЯЩЕМ шаге, по
образцу bench_prefill_rest_ab: плечо = набор заглушек, активных только на
трассировке своей скомпилированной функции; раунды чередуют плечи.

Правила заглушек (закон 34 и правило 17.09): та же форма и тот же dtype,
что у заменяемого; заглушка ЧИТАЕТ ВСЕ свои входы (pin), иначе MLX не
вычисляет питающий её подграф и меряется вырезанный кусок.

Замер декода -- как в bench_scales_1024: шаг по одному токену, argmax на
GPU, eval(tok, state) каждый токен. Образец плеча = K токенов подряд.

Плечи: full; gemv0 (все квантованные линейные слои, включая фьюз r/k/v и
голову -> нули); lora0; wkv0; ln_id; tail_id; prewkv_id; all (всё сразу);
sync (пустой скомпилированный шаг: пол синка на токен).

Контроль: отпечаток логитов фиксированной 4-шаговой цепочки на трассировке
и в конце обязан совпасть; заглушка обязана менять отпечаток; all >= sync.

Запуск: RWKVQ_COMP=<файл в WKV-kvant> python bench_decode_rest_ab.py
"""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
import rwkv_quant.backends.metal.quant_linear_sym as sy
from rwkv_quant.formats.reader import load_raw

KV = "/Users/s/Develop/WKV-kvant/"
COMP = os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")
if not COMP.startswith("/"):
    COMP = KV + COMP
K = int(os.environ.get("RWKVQ_K", "32"))
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "9"))
WARM = 2


def pin(*arrs):
    s = None
    for a in arrs:
        if a is None:
            continue
        e = a.reshape(-1)[:1].astype(mx.float32)
        s = e if s is None else s + e
    return s * 0.0


def _lin_stub(self, x):
    return mx.zeros(x.shape[:-1] + (self.out_features,), mx.float32) + pin(x)


def _fused_stub(self, xstack):
    return mx.zeros((self.K, self.out_per), mx.float32) + pin(xstack)


ORIG = {"wkv": qm._wkv_stateful, "ln": qm._layer_norm, "tail": qm.wkv_tail,
        "lora": qm.QuantTMix._lora, "prewkv": qm.QuantTMix._prewkv,
        "gl": gw.GwQuantLinear.__call__, "gf": gw.GwQuantLinearFused.__call__,
        "sl": sy.SymQuantLinear.__call__, "sf": sy.SymQuantLinearFused.__call__,
        "fprewkv": qm.FUSE_PREWKV}


def apply(names):
    for n in names:
        if n == "wkv":
            qm._wkv_stateful = lambda r, w, k, v, a, b, state: (
                mx.zeros(r.shape, mx.float32) + pin(r, w, k, v, a, b), state + pin(r, w, k, v, a, b))
        elif n == "ln":
            qm._layer_norm = lambda x, weight, bias, eps=1e-5, fast=None: x
        elif n == "tail":
            qm.wkv_tail = lambda wkv, r, k, v, r_k, ln_w, ln_b, g, H, S, eps=64e-5: (
                wkv.reshape(-1, H * S) + pin(r, k, v, g))
        elif n == "lora":
            qm.QuantTMix._lora = lambda self, xw, xa, xv, xg, x, xx: (
                (mx.zeros_like(x) + pin(xw, xa, xv, xg, xx).astype(x.dtype),) * 4)
        elif n == "prewkv":
            qm.QuantTMix._prewkv = lambda self, y_w, y_a, y_v, k, v, v_first, B, T, dtype: (
                k + pin(y_w, y_a, y_v, v_first).astype(k.dtype), k, v, k, k, v)
        elif n == "fprewkv":
            qm.FUSE_PREWKV = True
        elif n == "gemv":
            gw.GwQuantLinear.__call__ = _lin_stub
            sy.SymQuantLinear.__call__ = _lin_stub
            gw.GwQuantLinearFused.__call__ = _fused_stub
            sy.SymQuantLinearFused.__call__ = _fused_stub


def restore():
    qm._wkv_stateful = ORIG["wkv"]; qm._layer_norm = ORIG["ln"]; qm.wkv_tail = ORIG["tail"]
    qm.QuantTMix._lora = ORIG["lora"]; qm.QuantTMix._prewkv = ORIG["prewkv"]
    gw.GwQuantLinear.__call__ = ORIG["gl"]; gw.GwQuantLinearFused.__call__ = ORIG["gf"]
    sy.SymQuantLinear.__call__ = ORIG["sl"]; sy.SymQuantLinearFused.__call__ = ORIG["sf"]
    qm.FUSE_PREWKV = ORIG["fprewkv"]


ARMS = [("full", []), ("gemv0", ["gemv"]), ("lora0", ["lora"]), ("wkv0", ["wkv"]),
        ("ln_id", ["ln"]), ("tail_id", ["tail"]), ("prewkv_id", ["prewkv"]),
        ("all", ["gemv", "lora", "wkv", "ln", "tail", "prewkv"]),
        # готовое ядро пред-WKV (11.09, выключено) -- НЕ заглушка, отпечаток может совпасть с full
        ("prewkv_kern", ["fprewkv"])]
SEL = os.environ.get("RWKVQ_ARMS")
if SEL:
    ARMS = [a for a in ARMS if a[0] in SEL.split(",")]
else:
    ARMS = [a for a in ARMS if a[0] != "prewkv_kern"]


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


sw0 = swap_used()
print("своп до: %.1f МБ; %s K=%d раундов %d" % (sw0, os.path.basename(COMP), K, ROUNDS), flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
# линейные слои модели по типам -- чтобы видеть, что gemv0 накрыл все
kinds = {}
for b in m.blocks:
    for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj, b.cmix.key, b.cmix.value):
        kinds[type(l).__name__] = kinds.get(type(l).__name__, 0) + 1
kinds["head:" + type(m.head).__name__] = 1
print("  линейные слои:", kinds, flush=True)
# eager-шаг: строит ленивые квантованные копии LoRA и фьюз ВНЕ трассировки
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1))
mx.eval(lg)
print("  rkv_fused:", type(m.blocks[1].tmix._rkv_fused).__name__, flush=True)

FIX = [mx.array([[t]], dtype=mx.int32) for t in (11, 222, 3333, 44444)]


def fingerprint(fn):
    st = m.init_state(1)
    for t in FIX:
        lg, st = fn(t, st)
    return float(mx.sum(mx.abs(lg.astype(mx.float32))).item())


FNS, FP0, ST = {}, {}, {}
for name, stubs in ARMS:
    apply(stubs)
    fn = mx.compile(lambda i, s: m.forward_stateful(i, s))
    FP0[name] = fingerprint(fn)
    st = m.init_state(1)
    tok = mx.array([7], dtype=mx.int32)
    for _ in range(WARM * 4):
        lg, st = fn(tok[None], st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok, st)
    restore()
    FNS[name], ST[name] = fn, (st, tok)
    print("  трассировано %-10s отпечаток %.6e" % (name, FP0[name]), flush=True)

# пол синка: скомпилированный тривиальный шаг с тем же ритмом eval
triv = mx.compile(lambda t: (t + 1) % 65536)
mx.eval(triv(mx.array([7], dtype=mx.int32)))  # трассировка вне замера
V = {n: [] for n, _ in ARMS}
V["sync"] = []


def run_arm(name):
    fn = FNS[name]
    st, tok = ST[name]
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(K):
        lg, st = fn(tok[None], st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok, st)
    mx.synchronize()
    dt = (time.perf_counter() - t0) * 1e3 / K
    ST[name] = (st, tok)
    return dt


def run_sync():
    t = mx.array([7], dtype=mx.int32)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(K):
        t = triv(t)
        mx.eval(t)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3 / K


names = [n for n, _ in ARMS] + ["sync"]
for rd in range(ROUNDS):
    order = names if rd % 2 == 0 else names[::-1]
    for name in order:
        V[name].append(run_sync() if name == "sync" else run_arm(name))

ok = True
for name, _ in ARMS:
    fp = fingerprint(FNS[name])
    same = fp == FP0[name] or (np.isnan(fp) and np.isnan(FP0[name]))
    if not same:
        ok = False
        print("  НЕДЕЙСТВИТЕЛЬНО: %s отпечаток уехал %.6e -> %.6e" % (name, FP0[name], fp), flush=True)
for name, _ in ARMS[1:]:
    if name != "prewkv_kern" and FP0[name] == FP0.get("full"):
        ok = False
        print("  НЕДЕЙСТВИТЕЛЬНО: заглушка %s не изменила выход" % name, flush=True)

med = {n: float(np.median(V[n])) for n in names}
full = med.get("full", float("nan"))
print("--- медианы, мс/ток (разброс = (max-min)/медиана), статья = full - плечо ---", flush=True)
tot = 0.0
for name in names:
    d = full - med[name]
    if name not in ("full", "all", "sync", "prewkv_kern"):
        tot += d
    print("  %-10s %7.3f  разброс %4.1f%%  статья %+7.3f  (%5.1f%% шага)" % (
        name, med[name], 100 * (max(V[name]) - min(V[name])) / med[name], d, 100 * d / full), flush=True)
if "all" in med:
    print("  сумма статей по одной: %.3f; full - all: %.3f (разница = взаимодействие)" % (tot, full - med["all"]), flush=True)
    print("  остаток all (эмб, сдвиги, лерпы, активация cmix, резидуал, argmax, склейка запусков): %.3f; из них синк %.3f" % (med["all"], med["sync"]), flush=True)
    if med["all"] < med["sync"]:
        ok = False
        print("  НЕДЕЙСТВИТЕЛЬНО: all ниже пола синка -- граф вырезан", flush=True)
print("  ток/с full: %.2f; своп %+.1f МБ; активная %.0f МБ; %s" % (
    1000 / full, swap_used() - sw0, mx.get_active_memory() / 1e6, "ДЕЙСТВИТЕЛЬНО" if ok else "НЕДЕЙСТВИТЕЛЬНО"), flush=True)
