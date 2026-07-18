"""Пер-компонентный профиль decode чемпиона v2 (/tmp/champion_v2.rwkvq).
Метод: аблации (подмена компонента заглушкой) с A/B-ЧЕРЕДОВАНИЕМ в одном
процессе (закон №1: безвентиляторный дрейф). Каждая аблация: R раундов
[full -> ablated], дельта = медиана поразрядных разностей.
Сырой forward_stateful (не mx.compile) -- monkeypatch невидим компилятору.
Отдельно: compiled step vs raw, для оценки launch/graph overhead."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

model = qm.QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
N_EMBD = 2048

def timed(n=20, warm=3):
    states = timed.states
    idx = mx.array(np.array([[123]], dtype=np.int64))
    for _ in range(warm):
        logits, states = model.forward_stateful(idx, states)
        mx.eval(logits, *[s for st in states for s in st if s is not None])
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, states = model.forward_stateful(idx, states)
        mx.eval(logits, *[s for st in states for s in st if s is not None])
    mx.synchronize()
    timed.states = states
    return (time.perf_counter() - t0) / n * 1e3
timed.states = model.init_state(1)

def ab_delta(name, apply, revert, R=6):
    deltas, fulls = [], []
    for _ in range(R):
        tA = timed()
        apply()
        tB = timed()
        revert()
        fulls.append(tA); deltas.append(tA - tB)
    print(f"{name:28s} delta={np.median(deltas):6.2f} ms  "
          f"(full med {np.median(fulls):5.2f}, spread {min(fulls):.2f}-{max(fulls):.2f})",
          flush=True)
    return np.median(deltas), np.median(fulls)

print("=== raw forward_stateful, аблации A/B ===", flush=True)
t_full0 = timed(n=30, warm=8)
print(f"full raw (первичный):        {t_full0:6.2f} ms/tok", flush=True)

# 1. WKV
_orig_wkv = qm._wkv_stateful
d_wkv, _ = ab_delta("WKV -> passthrough",
    lambda: setattr(qm, "_wkv_stateful", lambda r,w,k,v,a,b,st: (v, st)),
    lambda: setattr(qm, "_wkv_stateful", _orig_wkv))

# 2. head
_orig_head = model.head
class TinyHead:
    def __call__(self, x): return x[..., :16]
d_head, _ = ab_delta("head -> tiny",
    lambda: setattr(model, "head", TinyHead()),
    lambda: setattr(model, "head", _orig_head))

# 3. LoRA g/a/w/v -> rank-1 нули
_saved = []
def lora_off():
    D = N_EMBD
    for blk in model.blocks:
        tm = blk.tmix
        _saved.append((tm.g_lora_A, tm.g_lora_B_w, tm.a_lora_A, tm.a_lora_B_w,
                       tm.w_lora_A, tm.w_lora_B_w,
                       getattr(tm, "v_lora_A", None), getattr(tm, "v_lora_B_w", None)))
        tm.g_lora_A = mx.zeros((1, D)); tm.g_lora_B_w = mx.zeros((D, 1))
        tm.a_lora_A = mx.zeros((1, D)); tm.a_lora_B_w = mx.zeros((D, 1))
        tm.w_lora_A = mx.zeros((1, D)); tm.w_lora_B_w = mx.zeros((D, 1))
        if getattr(tm, "v_lora_A", None) is not None:
            tm.v_lora_A = mx.zeros((1, D)); tm.v_lora_B_w = mx.zeros((D, 1))
def lora_on():
    for blk, s in zip(model.blocks, _saved):
        tm = blk.tmix
        (tm.g_lora_A, tm.g_lora_B_w, tm.a_lora_A, tm.a_lora_B_w,
         tm.w_lora_A, tm.w_lora_B_w, vA, vB) = s
        if vA is not None: tm.v_lora_A, tm.v_lora_B_w = vA, vB
    _saved.clear()
d_lora, _ = ab_delta("LoRA -> rank-1", lora_off, lora_on)

# 4. cmix -> ноль (убирает оба GEMV key/value + relu^2 + shift)
_saved_cm = []
def cmix_off():
    for blk in model.blocks:
        _saved_cm.append(blk.cmix)
        class ZeroCM:
            def forward_stateful(self, x, ss): return x * 0.0, ss
        blk.cmix = ZeroCM()
def cmix_on():
    for blk, cm in zip(model.blocks, _saved_cm): blk.cmix = cm
    _saved_cm.clear()
d_cmix, _ = ab_delta("cmix -> zero", cmix_off, cmix_on)

# 5. tmix-проекции r/k/v/o -> о стоимости судим по остатку

t_full1 = timed(n=30)
print(f"full raw (контроль дрейфа):  {t_full1:6.2f} ms/tok", flush=True)

resid = np.median([t_full0, t_full1]) - d_wkv - d_head - d_lora - d_cmix
print(f"\nWKV {d_wkv:.2f} | head {d_head:.2f} | LoRA {d_lora:.2f} | cmix {d_cmix:.2f} "
      f"| остаток (tmix r/k/v/o GEMV + shift/LN/обвязка) {resid:.2f} ms", flush=True)

# === compiled step vs raw (A/B тоже чередуем) ===
print("\n=== compiled step vs raw ===", flush=True)
def timed_compiled(n=20, warm=8):
    st = model.init_state(1)
    idx = mx.array(np.array([[123]], dtype=np.int64))
    logits, st = model.step(idx, st)
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(warm):
        logits, st = model.step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = model.step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

pairs = [(timed_compiled(), timed()) for _ in range(4)]
tc = np.median([p[0] for p in pairs]); tr = np.median([p[1] for p in pairs])
print(f"compiled step: {tc:6.2f} ms/tok | raw: {tr:6.2f} ms/tok | выигрыш compile {tr-tc:.2f} ms", flush=True)
