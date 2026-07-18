"""
Эмпирический потолок FP16-компьюта на этой M4 + сравнение с реальными
FLOP-нагрузками decode-пути (proj/cmix/LoRA/head), чтобы понять, где
"остаток" (WKV+LoRA+shift+launch, ~5.5-8.3 мс из профиля 19.07/19.07-2)
упирается: в ALU, в bandwidth или в launch/occupancy тиковых матмулов.

Методология как в bench_kernel_clean.py: хост-синк меряется отдельно и
вычитается; несколько повторов, median; несколько reps в ОДНОМ eval,
чтобы амортизировать sync (закон 19.07-2: изолированный eval+sync сам
стоит ~0.21 мс и искажает мелкие кернели).

Три режима:
  A. ceiling  -- большой квадратный fp16 матмул (compute-bound по
     построению: N=2048/4096, arithmetic intensity высокая), несколько
     ЗАВИСИМЫХ матмулов в одном eval чтобы GPU не простаивал на sync.
  B. real-shape isolated -- те же формы, что реально гоняются в decode
     (proj 2048x2048 GEMV, cmix 8192x2048 GEMV, LoRA A/B по рангам
     w=96/a=96/v=64/g=256, head 65536x2048 GEMV), batch=1 (T=1, как в
     decode), много НЕЗАВИСИМЫХ повторов в одном eval.
  C. LoRA batched -- одним матмулом на все 24 слоя сразу (как ceiling),
     чтобы увидеть достижимый GFLOPS ПРИ ЭТОЙ форме, если бы её грузить
     оптимально, а не 24 отдельными вызовами.

Вывод: GFLOPS(B) / GFLOPS(A) -- какая доля потолка используется в
реальной decode-форме. Если << 1% -- дело не в ALU (что и подозревали),
и "менять математику" (уменьшать FLOP) бессмысленно; узкое место --
launch/occupancy отдельных крошечных вызовов, а не throughput.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mlx.core as mx

D = 2048          # n_embd (rwkv7-g1h-1.5b)
CMIX_HID = 8192   # ffn hidden
VOCAB = 65536
N_LAYER = 24
LORA_RANKS = {"w": 96, "a": 96, "v": 64, "g": 256}

def bench(fn, reps=10, warm=3):
    for _ in range(warm):
        r = fn(); mx.eval(r) if not isinstance(r, (list, tuple)) else mx.eval(*r)
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        if isinstance(r, (list, tuple)):
            mx.eval(*r)
        else:
            mx.eval(r)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))

def sync_only():
    x = mx.array([1.0], dtype=mx.float16)
    return x + 1.0

def rand16(*shape):
    a = mx.array(np.random.randn(*shape).astype(np.float32) * 0.02).astype(mx.float16)
    mx.eval(a)
    return a

print("=== 0. хост-синк ===")
sync_acc = [bench(sync_only, reps=30) for _ in range(5)]
sync = float(np.median(sync_acc))
print(f"sync overhead: {sync:.4f} ms")

# ---------------------------------------------------------------------
print("\n=== A. compute ceiling (большой compute-bound матмул) ===")
CEIL_N = 2048
CHAIN = 8  # зависимая цепочка матмулов в одном eval -> амортизирует sync,
           # не даёт GPU простаивать между вызовами

a0 = rand16(CEIL_N, CEIL_N)
b0 = rand16(CEIL_N, CEIL_N)

def ceiling_chain():
    x = a0
    for _ in range(CHAIN):
        x = mx.matmul(x, b0)
    return x

t_ceil = min(bench(ceiling_chain, reps=8, warm=3) for _ in range(3))
flops_ceil = 2 * (CEIL_N ** 3) * CHAIN
gflops_ceil = flops_ceil / ((t_ceil - sync) * 1e-3) / 1e9
print(f"N={CEIL_N} x{CHAIN} chained matmul: {t_ceil:.3f} ms (sync-corrected "
      f"{t_ceil - sync:.3f} ms) -> {gflops_ceil:,.0f} GFLOPS achieved ceiling")

# также на форме, близкой к cmix (8192x2048 @ 2048xN), N large для compute-bound
M2, K2, N2 = CMIX_HID, D, 512
w2 = rand16(M2, K2)
x2 = rand16(K2, N2)
def cmixshape_ceiling():
    return mx.matmul(w2, x2)
t_c2 = min(bench(cmixshape_ceiling, reps=15, warm=3) for _ in range(3))
flops_c2 = 2 * M2 * K2 * N2
gflops_c2 = flops_c2 / ((t_c2 - sync) * 1e-3) / 1e9
print(f"cmix-shape ({M2}x{K2} @ {K2}x{N2}): {t_c2:.3f} ms -> {gflops_c2:,.0f} GFLOPS")

# ---------------------------------------------------------------------
print("\n=== B. реальные формы decode (batch=1, T=1), ИЗОЛИРОВАННО ===")

def gemv_bench(name, M, K, reps=30):
    w = rand16(M, K)
    x = rand16(1, K)
    def f(): return mx.matmul(x, w.T)
    t = min(bench(f, reps=reps, warm=5) for _ in range(3))
    flops = 2 * M * K
    gflops = flops / max((t - sync), 1e-6) / 1e-3 / 1e9
    print(f"{name:22s} {M}x{K} GEMV: {t:7.4f} ms -> {gflops:8.2f} GFLOPS "
          f"({gflops/gflops_ceil*100:5.2f}% потолка A)")
    return t, flops

gemv_bench("proj (r/k/v/o)", D, D)
gemv_bench("cmix.key", CMIX_HID, D)
gemv_bench("cmix.value", D, CMIX_HID)
gemv_bench("head", VOCAB, D, reps=10)

print("\n-- LoRA A/B по рангам, batch=1, ИЗОЛИРОВАННО --")
lora_flops_per_layer = 0
for name, r in LORA_RANKS.items():
    wA = rand16(r, D); wB = rand16(D, r)
    x = rand16(1, D)
    def f(wA=wA, wB=wB, x=x):
        h = mx.matmul(x, wA.T)
        return mx.matmul(h, wB.T)
    t = min(bench(f, reps=50, warm=10) for _ in range(3))
    flops = 4 * D * r  # down + up, 2*D*r each
    lora_flops_per_layer += flops
    gflops = flops / max((t - sync), 1e-6) / 1e-3 / 1e9
    print(f"lora.{name:2s} (r={r:3d})       {t:7.4f} ms -> {gflops:8.4f} GFLOPS "
          f"({gflops/gflops_ceil*100:6.3f}% потолка A)")

print(f"\nLoRA FLOPs/layer = {lora_flops_per_layer:,} ; x{N_LAYER} слоёв = "
      f"{lora_flops_per_layer*N_LAYER:,} FLOPs total")

# ---------------------------------------------------------------------
print("\n=== C. LoRA batched across all 24 layers (одним матмулом) ===")
for name, r in LORA_RANKS.items():
    wA = rand16(N_LAYER, r, D); wB = rand16(N_LAYER, D, r)
    x = rand16(N_LAYER, 1, D)
    def f(wA=wA, wB=wB, x=x):
        h = mx.matmul(x, wA.transpose(0, 2, 1))
        return mx.matmul(h, wB.transpose(0, 2, 1))
    t = min(bench(f, reps=30, warm=5) for _ in range(3))
    flops = 4 * D * r * N_LAYER
    gflops = flops / max((t - sync), 1e-6) / 1e-3 / 1e9
    print(f"lora.{name:2s} batched x{N_LAYER} (r={r:3d}) {t:7.4f} ms -> "
          f"{gflops:9.2f} GFLOPS ({gflops/gflops_ceil*100:5.2f}% потолка A)")

# ---------------------------------------------------------------------
print("\n=== Сводка FLOP-бюджета decode (1 токен, аналитика) ===")
proj_flops = 4 * 2 * D * D
cmix_flops = 2 * CMIX_HID * D + 2 * D * CMIX_HID
lora_total = lora_flops_per_layer
per_layer = proj_flops + cmix_flops + lora_total
total = per_layer * N_LAYER + 2 * VOCAB * D
print(f"proj/layer   = {proj_flops:>14,} FLOPs")
print(f"cmix/layer   = {cmix_flops:>14,} FLOPs")
print(f"lora/layer   = {lora_total:>14,} FLOPs")
print(f"per-layer    = {per_layer:>14,} FLOPs  x{N_LAYER} layers = {per_layer*N_LAYER:>16,}")
print(f"head (once)  = {2*VOCAB*D:>14,} FLOPs")
print(f"TOTAL/token  = {total:>14,} FLOPs  ({total/1e9:.3f} GFLOP/token)")
print(f"\nПри наблюдаемом 'GEMV-бюджете' decode ~12.5 мс (профиль 19.07):")
print(f"  implied throughput = {total/12.5e-3/1e9:.1f} GFLOPS "
      f"({total/12.5e-3/1e9/gflops_ceil*100:.2f}% потолка A)")
