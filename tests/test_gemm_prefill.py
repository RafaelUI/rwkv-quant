"""GEMM-путь префилла (N >= GEMM_MIN_BATCH: dequant fp16 + mx.matmul) в
QuantLinearV2: численный паритет с v1 (fp32-эталон) и бенч против GEMV.

Методология бенча: длинный прогрев (spin ~1.5с) до устоявшейся частоты GPU --
короткие микробенчи на Apple Silicon ловят случайные DVFS-состояния
(урок вопроса №4 в NEXT_SESSION.md)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import _real_quantize, _real_quantize_sparse_outlier
from rwkv_quant.formats.schema import QuantizedTensor, pack_int4
from rwkv_quant.backends.metal.quant_linear import QuantLinear
from rwkv_quant.backends.metal import quant_linear_v2 as qv2
from rwkv_quant.backends.metal.quant_linear_v2 import QuantLinearV2

torch.manual_seed(0); np.random.seed(0)

def make(OUT, IN, bits, spqr, packed):
    w = torch.randn(OUT, IN) * torch.exp(torch.randn(OUT, 1))
    if spqr:
        c, s, oi, ov = _real_quantize_sparse_outlier(w, bits, 0.02)
    else:
        (c, s), oi, ov = _real_quantize(w, bits), None, None
    cp = None
    if packed:
        assert bits <= 4
        cp, c = pack_int4(c), None
    return QuantizedTensor(key="t", group="proj", bits=bits, shape=(OUT, IN),
                           codes=c, codes_packed=cp, scale=s,
                           outlier_indices=oi, outlier_values=ov)

print("== паритет GEMM-путь (N=64) vs v1 fp32-эталон ==")
ok = True
for OUT, IN in [(2048, 2048), (8192, 2048), (2048, 8192), (768, 3072), (65536, 2048)]:
    for bits, packed in [(8, False), (4, False), (4, True)]:
        for spqr in (False, True):
            qt = make(OUT, IN, bits, spqr, packed)
            q1, q2 = QuantLinear(qt), QuantLinearV2(qt)
            x = mx.array(np.random.randn(64, IN).astype(np.float32))
            y1, y2 = q1(x), q2(x)
            assert q2.packed == packed
            rel = float(mx.abs(y1 - y2).max() / (mx.abs(y1).max() + 1e-9))
            # fp16-операнды GEMM: допуск как у fp16-dense пути (~1e-3),
            # GEMV-ветки остаются на прежнем допуске 1e-5 в test_quant_linear_v2
            status = "OK " if rel < 2e-3 else "FAIL"
            if rel >= 2e-3: ok = False
            if spqr:
                print(f"{status} {OUT}x{IN} bits={bits} packed={packed} spqr: rel {rel:.2e}")
assert ok, "GEMM-путь разошёлся с эталоном"

print("\n== порог: N=15 идёт через GEMV (fp32-точность), N=16 через GEMM ==")
qt = make(2048, 2048, 8, True, False)
q1, q2 = QuantLinear(qt), QuantLinearV2(qt)
x15 = mx.array(np.random.randn(15, 2048).astype(np.float32))
rel15 = float(mx.abs(q1(x15) - q2(x15)).max() / mx.abs(q1(x15)).max())
assert rel15 < 1e-5, f"N=15 должен идти GEMV-веткой, rel={rel15:.2e}"
print(f"OK  N=15 rel {rel15:.2e} (GEMV), порог не сдвинут")

print("\n== бенч: мс на вызов, T=256 (per-token в скобках) ==")
def spin(sec=1.5):
    a = mx.ones((2048, 2048), dtype=mx.float16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        mx.eval(a @ a)

def bench(fn, x, iters=30):
    # eval КАЖДУЮ итерацию: ленивое накопление 30 деквант-графов однажды
    # раздуло процесс до 15.8GB и ушло в своп (head: сотни MB транзиента
    # на вызов). Пиковая память = один вызов, не iters вызовов.
    for _ in range(3): mx.eval(fn(x))
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): mx.eval(fn(x))
    mx.synchronize(); return (time.perf_counter() - t0) / iters * 1e3

T = 256
print(f"{'shape':>14} {'cfg':>12} | {'GEMV':>9} | {'GEMM':>9} | speedup")
spin()
import gc
for OUT, IN in [(2048, 2048), (8192, 2048), (2048, 8192), (65536, 2048)]:
    for bits, packed, spqr in [(8, False, True), (4, True, True)]:
        qt = make(OUT, IN, bits, spqr, packed)
        q = QuantLinearV2(qt)
        x = mx.array(np.random.randn(T, IN).astype(np.float32))
        iters = 10 if OUT >= 65536 else 30
        saved = qv2.GEMM_MIN_BATCH
        try:
            qv2.GEMM_MIN_BATCH = 10**9   # форсируем GEMV
            tg = bench(q, x, iters)
            qv2.GEMM_MIN_BATCH = saved   # штатный GEMM-путь
            tm = bench(q, x, iters)
        finally:
            qv2.GEMM_MIN_BATCH = saved
        del q, qt; gc.collect(); mx.clear_cache()
        cfg = f"int{bits}{'p' if packed else ''}{'+spqr' if spqr else ''}"
        print(f"{OUT:>6}x{IN:<7} {cfg:>12} | {tg:8.3f} | {tm:8.3f} | {tg/tm:5.2f}x  ({tm/T*1e3:.1f} мкс/ток)")
print("\nвсё зелёное")
