"""Пер-компонентный профиль ПРЕФИЛЛА чемпиона v2 (/tmp/champion_v2.rwkvq),
T=1024. Аналог profile_decode_v2.py, но для forward_stateful(idx, state,
last_only=True) с T>1 (не T=1 decode).

ВАЖНО: старые bench_prefill.py / test_gemm_prefill.py / profile_target.py
меряют GEMM-путь QuantLinearV2 (v1-формат, /tmp/compression_packed.rwkvq).
Чемпион (18.07-вечер+) — gw sb6 формат, свой кернель GwQuantLinear со своим
GEMM-путём (_dequant_w + mx.matmul, GEMM_MIN_BATCH=16 в quant_linear_gw.py,
НЕЗАВИСИМ от quant_linear_v2.GEMM_MIN_BATCH). Этот gw-GEMM-путь для
префилла ни разу не профилировался отдельно -- это первый замер.

Метод: A/B-чередование в одном процессе (закон №1), R раундов [full ->
ablated], дельта = медиана поразрядных разностей.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as qgw

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
CKPT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/champion_v2.rwkvq"

model = qm.QuantRWKV7(load_raw(CKPT))
N_EMBD = model.n_embd
idx = mx.array(np.random.randint(0, 65000, (1, T)).astype(np.int64))


def _flat(st):
    return [s for x in st for s in x if s is not None]


def timed(n=5, warm=2):
    for _ in range(warm):
        logits, st = model.forward_stateful(idx, model.init_state(1), last_only=True)
        mx.eval(logits, *_flat(st))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = model.forward_stateful(idx, model.init_state(1), last_only=True)
        mx.eval(logits, *_flat(st))
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3  # ms per full prefill


def ab_delta(name, apply, revert, R=3, n=3):
    deltas, fulls = [], []
    for _ in range(R):
        tA = timed(n=n)
        apply()
        tB = timed(n=n)
        revert()
        fulls.append(tA); deltas.append(tA - tB)
    print(f"{name:28s} delta={np.median(deltas):7.1f} ms  "
          f"(full med {np.median(fulls):7.1f}, spread {min(fulls):.1f}-{max(fulls):.1f})  "
          f"[{np.median(deltas)/T*1e3:.3f} мкс/ток]",
          flush=True)
    return np.median(deltas), np.median(fulls)


def spin(sec=2.0):
    a = mx.ones((2048, 2048), dtype=mx.float16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        mx.eval(a @ a)


print(f"PID {os.getpid()}  ckpt={CKPT} T={T}", flush=True)
spin()

print("=== raw forward_stateful (last_only=True), аблации A/B ===", flush=True)
t_full0 = timed(n=5, warm=3)
print(f"full raw (первичный):        {t_full0:8.1f} ms  ({t_full0/T:.3f} ms/tok)", flush=True)

# 1. WKV -> passthrough
_orig_wkv = qm._wkv_stateful
d_wkv, _ = ab_delta("WKV -> passthrough",
    lambda: setattr(qm, "_wkv_stateful", lambda r, w, k, v, a, b, st: (v, st)),
    lambda: setattr(qm, "_wkv_stateful", _orig_wkv))

# 2. head -> tiny (last_only=True уже режет до 1 позиции, но матрица всё равно полная)
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

# 4. cmix -> ноль (proj key/value GEMM большие -- 805M эл, самая тяжёлая группа)
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

t_full1 = timed(n=5)
print(f"full raw (контроль дрейфа):  {t_full1:8.1f} ms  ({t_full1/T:.3f} ms/tok)", flush=True)

resid = np.median([t_full0, t_full1]) - d_wkv - d_head - d_lora - d_cmix
print(f"\nWKV {d_wkv:.1f} | head {d_head:.1f} | LoRA {d_lora:.1f} | cmix {d_cmix:.1f} "
      f"| остаток (tmix r/k/v/o GEMM + shift/LN/обвязка) {resid:.1f} ms  "
      f"[{resid/T*1e3:.3f} мкс/ток]", flush=True)

# === GEMM vs GEMV путь В gw-кернеле (независимо от старого qv2.GEMM_MIN_BATCH) ===
print("\n=== gw-кернель: GEMM vs GEMV путь (qgw.GEMM_MIN_BATCH) ===", flush=True)
saved_gmb = qgw.GEMM_MIN_BATCH
qgw.GEMM_MIN_BATCH = 10**9
t_gemv = timed(n=3, warm=2)
qgw.GEMM_MIN_BATCH = saved_gmb
t_gemm = timed(n=5, warm=2)
print(f"GEMV-путь (форс): {t_gemv:8.1f} ms ({t_gemv/T:.3f} ms/tok)  |  "
      f"GEMM-путь (штатно): {t_gemm:8.1f} ms ({t_gemm/T:.3f} ms/tok)  |  "
      f"speedup {t_gemv/t_gemm:.2f}x", flush=True)
