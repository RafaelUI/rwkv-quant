"""РАЗЛОЖЕНИЕ ПРЕФИЛЛА В ОДНОМ ПРОЦЕССЕ (17.09): куда уходят ~125 мс сверх
матмулов и декванта. Путь рабочий: m.forward_stateful(last_only=True) под
mx.compile, как m.step в bench_prefill_deqkern_ab. Плечо = набор заглушек,
активных ТОЛЬКО на трассировке своей скомпилированной функции; раунды
чередуют плечи (закон про дрейф безвентиляторной машины).

Заглушки не меняют форм, только значения (нули/тождество); время кернелей
от значений не зависит. Контроль ПОДМЕНЫ ГРАФА: отпечаток каждого плеча на
трассировке и в конце обязан совпасть (иначе compile перетрассировал без
заглушки), а плечо deq_cache обязано дать отпечаток ПОЛНОГО (кеш -- та же
цепочка, кернель ей бит-в-бит равен).

Плечи: full; wkv0 (скан -> нули, состояние не трогается); lora0 (все LoRA
-> нули); ln_id (_layer_norm -> тождество); tail_id (group_norm+bonus+gate
-> тождество); prewkv_id (пред-WKV -> перестановка входов); deq_cache
(плотные веса заранее, пол для статьи декванта); all (всё сразу: остаток =
матмулы + лерпы + активация cmix + эмбеддинг + касты).
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
WARM = 2

ORIG = {"wkv": qm._wkv_stateful, "ln": qm._layer_norm, "tail": qm.wkv_tail,
        "lora": qm.QuantTMix._lora, "prewkv": qm.QuantTMix._prewkv,
        "deq": gw.GwQuantLinear._dequant_w, "lora_q": qm.LORA_Q}
_CACHE = {}


def _deq_cached(self):
    w = _CACHE.get(id(self))
    if w is None:
        w = gw.GwQuantLinear._dequant_w_ref(self)
        mx.eval(w)
        _CACHE[id(self)] = w
    return w


# ПРАВКА ПОСЛЕ ПЕРВОГО ПРОГОНА: MLX ленив, и выход, от которого ничего не
# зависит, НЕ ВЫЧИСЛЯЕТСЯ. Заглушка скана, не читавшая w/k/v/a/b, выключала
# вместе со сканом k_proj, v_proj, пред-WKV и LoRA w/a/v -- статья wkv0 была
# их суммой, а all провалился НИЖЕ пола матмулов (374 мс). Поэтому каждая
# заглушка ПОТРЕБЛЯЕТ все свои входы: первый элемент каждого, умноженный на
# ноль. Срез одного элемента требует вычисления всего источника (срез не
# проталкивается внутрь матмула), а стоит сам по себе ничего.
def pin(*arrs):
    s = None
    for a in arrs:
        if a is None:
            continue
        e = a.reshape(-1)[:1].astype(mx.float32)
        s = e if s is None else s + e
    return s * 0.0


STUB = {
    "wkv": ("wkv", lambda r, w, k, v, a, b, state: (mx.zeros(r.shape, mx.float32) + pin(r, w, k, v, a, b), state)),
    "ln": ("ln", lambda x, weight, bias, eps=1e-5, fast=None: x),
    "tail": ("tail", lambda wkv, r, k, v, r_k, ln_w, ln_b, g, H, S, eps=64e-5: wkv + pin(r, k, v, g)),
    "lora": ("lora", lambda self, xw, xa, xv, xg, x, xx: (mx.zeros_like(x) + pin(xw, xa, xv, xg, xx).astype(x.dtype),) * 4),
    "prewkv": ("prewkv", lambda self, y_w, y_a, y_v, k, v, v_first, B, T, dtype: (k + pin(y_w, y_a, y_v, v_first).astype(k.dtype), k, v, k, k, v)),
    "deq": ("deq", _deq_cached),
    "wav": ("wav", None),
}


def apply(names):
    for n in names:
        f = STUB[n][1]
        if n == "wkv": qm._wkv_stateful = f
        elif n == "ln": qm._layer_norm = f
        elif n == "tail": qm.wkv_tail = f
        elif n == "lora": qm.QuantTMix._lora = f
        elif n == "prewkv": qm.QuantTMix._prewkv = f
        elif n == "deq": gw.GwQuantLinear._dequant_w = f
        elif n == "wav": qm.LORA_Q = None


def restore():
    qm._wkv_stateful = ORIG["wkv"]; qm._layer_norm = ORIG["ln"]; qm.wkv_tail = ORIG["tail"]
    qm.QuantTMix._lora = ORIG["lora"]; qm.QuantTMix._prewkv = ORIG["prewkv"]
    gw.GwQuantLinear._dequant_w = ORIG["deq"]
    qm.LORA_Q = ORIG["lora_q"]


ARMS = [("full", []), ("wkv0", ["wkv"]), ("lora0", ["lora"]), ("ln_id", ["ln"]),
        ("tail_id", ["tail"]), ("prewkv_id", ["prewkv"]), ("deq_cache", ["deq"]),
        ("all", ["wkv", "lora", "ln", "tail", "prewkv", "deq"]),
        # 17.09 ночь: батченый путь LoRA (w,a,v двумя матмулами по стекам _wav_At)
        # на префилле -- LORA_Q=None на трассировке. НЕ бит-в-бит (другие формы
        # матмула), поэтому отпечаток сверяется только с самим собой.
        ("lora_wav", ["wav"])]
SEL = os.environ.get("RWKVQ_ARMS")
if SEL:
    ARMS = [a for a in ARMS if a[0] in SEL.split(",")]


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


sw0 = swap_used()
print("своп до: %.1f МБ; %s T=%d раундов %d" % (sw0, COMP, T, ROUNDS), flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
IDX = mx.array(np.random.default_rng(0).integers(1, 60000, size=(1, T)).astype(np.int32))
FNS, FP0, LG = {}, {}, {}


def call(fn):
    lg, _ = fn(IDX, m.init_state(1))
    return lg


for name, st in ARMS:
    apply(st)
    fn = mx.compile(lambda i, s: m.forward_stateful(i, s, True))
    for _ in range(WARM):
        mx.eval(call(fn))
    lg = call(fn)
    LG[name] = lg
    mx.eval(lg)
    print("  активная память после %s: %.1f МБ" % (name, mx.get_active_memory() / 1e6), flush=True)
    FP0[name] = float(mx.sum(mx.abs(lg.astype(mx.float32))).item())
    restore()
    FNS[name] = fn
    print("  трассировано %-10s отпечаток %.6e" % (name, FP0[name]), flush=True)

V = {n: [] for n, _ in ARMS}
for rd in range(ROUNDS):
    order = ARMS if rd % 2 == 0 else ARMS[::-1]
    for name, _ in order:
        fn = FNS[name]
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(call(fn))
        mx.synchronize()
        V[name].append((time.perf_counter() - t0) * 1e3)

ok = True
for name, _ in ARMS:
    fp = float(mx.sum(mx.abs(call(FNS[name]).astype(mx.float32))).item())
    same = fp == FP0[name] or (np.isnan(fp) and np.isnan(FP0[name]))
    if not same:
        ok = False
        print("  НЕДЕЙСТВИТЕЛЬНО: %s отпечаток уехал %.6e -> %.6e" % (name, FP0[name], fp), flush=True)
if "deq_cache" in FP0 and FP0["deq_cache"] != FP0["full"]:
    ok = False
    print("  НЕДЕЙСТВИТЕЛЬНО: deq_cache не равен full (%.6e против %.6e)" % (FP0["deq_cache"], FP0["full"]), flush=True)
for name, st in ARMS[1:]:
    if name not in ("deq_cache", "lora_wav") and FP0[name] == FP0["full"]:
        ok = False
        print("  НЕДЕЙСТВИТЕЛЬНО: заглушка %s не изменила выход -- граф без неё" % name, flush=True)

full = float(np.median(V["full"]))
print("--- медианы, мс (разброс = (max-min)/медиана) ---", flush=True)
tot = 0.0
for name, _ in ARMS:
    med = float(np.median(V[name]))
    d = full - med
    if name not in ("full", "all"):
        tot += d
    print("  %-10s %7.1f  разброс %4.1f%%  статья %+7.1f" % (name, med, 100 * (max(V[name]) - min(V[name])) / med, d), flush=True)
allm = float(np.median(V["all"])) if "all" in V else float("nan")
if "lora_wav" in LG:
    a, b = LG["full"].astype(mx.float32), LG["lora_wav"].astype(mx.float32)
    print("  lora_wav против full: max|dlogit| %.3e, rel %.3e, argmax %s" % (mx.max(mx.abs(a - b)).item(), (mx.max(mx.abs(a - b)) / mx.max(mx.abs(a))).item(), "совпал" if mx.argmax(a).item() == mx.argmax(b).item() else "РАЗОШЁЛСЯ"), flush=True)
print("  сумма статей по одной: %.1f; full - all: %.1f (разница = взаимодействие)" % (tot, full - allm), flush=True)
print("  остаток all (матмулы по кешу + лерпы + cmix + emb + касты): %.1f" % allm, flush=True)
print("  своп %+.1f МБ; %s" % (swap_used() - sw0, "ДЕЙСТВИТЕЛЬНО" if ok else "НЕДЕЙСТВИТЕЛЬНО"), flush=True)
