import sys, time, re
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats import reader
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear


def dequant_w_precise(gw: GwQuantLinear) -> mx.array:
    """Как _dequant_w(), но финальный combine в float32 (бит-в-бит с
    reader._dequantize_gw_sb6): q*scale+mn в fp32, каст в bf16 один раз в конце."""
    OUT, IN = gw.out_features, gw.in_features
    cb = gw.codes.reshape(OUT, gw.NB, 16)
    q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.float32)
    if gw.xbits >= 1:
        bits = (gw.qh[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        bits = bits.reshape(OUT, IN).reshape(OUT, gw.NB, 32)
        q = q + bits.astype(mx.float32) * 16.0
    if gw.xbits >= 2:
        bits2 = (gw.qh2[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
        bits2 = bits2.reshape(OUT, IN).reshape(OUT, gw.NB, 32)
        q = q + bits2.astype(mx.float32) * 32.0
    s = (gw.qs.astype(mx.float32).reshape(OUT, gw.NSB, 8)
         * gw.d.astype(mx.float32)[..., None]).astype(mx.float16).astype(mx.float32)
    s = mx.maximum(s, 1e-8)
    m = (gw.qm.astype(mx.float32).reshape(OUT, gw.NSB, 8)
         * gw.dm.astype(mx.float32)[..., None]).astype(mx.float16).astype(mx.float32)
    w = q * s.reshape(OUT, gw.NB, 1) + m.reshape(OUT, gw.NB, 1)
    return w.reshape(OUT, IN).astype(mx.bfloat16)


ckpt = reader.load_raw("/tmp/reduction_v2.rwkvq")
targets = {}
for k, qt in ckpt.tensors.items():
    if not k.startswith("blocks.0.") and k not in ("emb.weight", "head.weight"):
        continue
    if qt.gw_mode != "sb6":
        continue
    generic = re.sub(r"blocks\.\d+\.", "blocks.N.", k)
    targets[generic] = (k, qt)

print(f"{'tensor':30s} {'mismatch':16s} {'warm_ms':8s}")
total = 0.0
for generic, (k, qt) in sorted(targets.items()):
    w_ref = reader._dequantize_gw_sb6(qt)
    gw = GwQuantLinear(qt)
    w = dequant_w_precise(gw)
    mx.eval(w)

    ref_bits = w_ref.view(torch.int16).numpy()
    got_bits = np.array(w.view(mx.uint16)).astype(np.int16)
    n_mis = int((ref_bits != got_bits).sum())

    N = 20
    t0 = time.time()
    for _ in range(N):
        w = dequant_w_precise(gw)
        mx.eval(w)
    dt = (time.time() - t0) / N * 1000
    total += dt if generic.startswith("blocks") else 0.0
    print(f"{generic:30s} {n_mis:8d}/{w_ref.numel():<10d} {dt:8.3f}")

print(f"\nsum per layer (proj+cmix) = {total:.2f} ms -> x24 = {total*24:.1f} ms")
