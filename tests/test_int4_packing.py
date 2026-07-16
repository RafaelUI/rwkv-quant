"""INT4 bit-packing: формат + packed-кернель.
1. quantize_tensor при bits<=4 даёт codes_packed, round-trip к int8 точен.
2. QuantLinearV2 на packed побитово?/численно == v1 (который распаковывает).
3. Размер: codes_packed вдвое меньше int8.
4. Бенч packed vs unpacked int8-кернель."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor, _real_quantize, _real_quantize_sparse_outlier
from rwkv_quant.formats.schema import QuantizedTensor, pack_int4, int8_codes
from rwkv_quant.formats.reader import _dequantize_one
from rwkv_quant.backends.metal.quant_linear import QuantLinear
from rwkv_quant.backends.metal.quant_linear_v2 import QuantLinearV2

torch.manual_seed(0); np.random.seed(0)
ok = True

# --- 1. writer пакует, dequant совпадает с некованым путём ---
w = torch.randn(256, 512)
cfg = QuantConfig(proj=4)
qt = quantize_tensor("blocks.0.tmix.r_proj.weight", w, cfg)
assert qt.codes is None and qt.codes_packed is not None, "bits=4 должен паковаться"
assert qt.codes_packed.shape == (256, 256) and qt.codes_packed.dtype == torch.uint8
codes_ref, scale_ref = _real_quantize(w, 4)
assert torch.equal(int8_codes(qt), codes_ref), "распаковка != исходные codes"
print("1. writer packs + round-trip: OK")

# --- 2. кернель: packed vs v1 ---
def mk(OUT, IN, spqr):
    w = torch.randn(OUT, IN) * torch.exp(torch.randn(OUT, 1))
    if spqr:
        c, s, oi, ov = _real_quantize_sparse_outlier(w, 4, 0.02)
    else:
        (c, s), oi, ov = _real_quantize(w, 4), None, None
    packed = QuantizedTensor(key="t", group="proj", bits=4, shape=(OUT,IN),
                             codes_packed=pack_int4(c), scale=s,
                             outlier_indices=oi, outlier_values=ov)
    plain  = QuantizedTensor(key="t", group="proj", bits=4, shape=(OUT,IN),
                             codes=c, scale=s, outlier_indices=oi, outlier_values=ov)
    return packed, plain

for OUT, IN in [(2048,2048),(8192,2048),(2048,8192),(768,3072),(65536,2048)]:
    for spqr in (False, True):
        qp, qu = mk(OUT, IN, spqr)
        q2p, q1 = QuantLinearV2(qp), QuantLinear(qu)
        assert q2p.packed, "packed-режим не включился"
        for N in (1, 4):
            x = mx.array(np.random.randn(N, IN).astype(np.float32))
            y1, y2 = q1(x), q2p(x)
            rel = float(mx.abs(y1-y2).max() / (mx.abs(y1).max()+1e-9))
            if rel >= 1e-5: ok = False; print(f"FAIL {OUT}x{IN} spqr={spqr} N={N}: {rel:.2e}")
print("2. packed-кернель vs v1: OK" if ok else "2. FAIL")

# --- 3. размер ---
q4 = quantize_tensor("blocks.0.tmix.r_proj.weight", torch.randn(2048,2048), QuantConfig(proj=4))
q8 = quantize_tensor("blocks.0.tmix.r_proj.weight", torch.randn(2048,2048), QuantConfig(proj=8))
b4 = q4.codes_packed.numel(); b8 = q8.codes.numel()
print(f"3. codes bytes 2048x2048: INT4 {b4/1e6:.2f}MB vs INT8 {b8/1e6:.2f}MB ({b8/b4:.1f}x): OK" if b8 == 2*b4 else "3. FAIL")
ok &= b8 == 2*b4

# --- 4. бенч ---
# ВАЖНО: короткий прогрев ловит GPU на низкой DVFS-частоте и даёт шум до 3x
# между прогонами (эмпирика M4). Прогреваем до рабочей частоты ~2с
# непрерывной нагрузкой, замер тоже длинный.
def bench(fn, x, warm_s=2.0, meas_s=1.5):
    t_end = time.perf_counter() + warm_s
    while time.perf_counter() < t_end:
        mx.eval([fn(x) for _ in range(100)])
    mx.synchronize(); t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < meas_s:
        mx.eval([fn(x) for _ in range(100)]); n += 100
    mx.synchronize(); return (time.perf_counter()-t0)/n*1e3
print("4. бенч (N=1):")
for OUT, IN in [(2048,2048),(8192,2048),(2048,8192),(65536,2048)]:
    qp, qu = mk(OUT, IN, False)
    q2p, q2u = QuantLinearV2(qp), QuantLinearV2(qu)
    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    tp, tu = bench(q2p, x), bench(q2u, x)
    print(f"   {OUT:>6}x{IN:<6} int8 {tu:.3f}ms -> packed {tp:.3f}ms ({tu/tp:.2f}x)")

print("\n[OK]" if ok else "\n[FAIL]"); sys.exit(0 if ok else 1)
