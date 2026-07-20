import sys, time, re
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats import reader
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear

ckpt = reader.load_raw("/tmp/reduction_v2.rwkvq")

targets = {}
for k, qt in ckpt.tensors.items():
    if not k.startswith("blocks.0.") and k not in ("emb.weight", "head.weight"):
        continue
    if qt.gw_mode != "sb6":
        continue
    generic = re.sub(r"blocks\.\d+\.", "blocks.N.", k)
    targets[generic] = (k, qt)

print(f"{len(targets)} форм\n")
print(f"{'tensor':30s} {'bits':4s} {'shape':18s} {'mismatch_bf16':14s} {'max_abs_diff':13s} {'warm_ms':8s}")

total = 0.0
for generic, (k, qt) in sorted(targets.items()):
    w_ref = reader._dequantize_gw_sb6(qt)  # bf16 torch reference

    gw = GwQuantLinear(qt)
    w = gw._dequant_w()  # fp16 mx.array, transient
    mx.eval(w)

    ref32 = w_ref.float().numpy()
    got32 = np.array(w.astype(mx.float32))
    diff = np.abs(ref32 - got32)
    # сравним после округления обоих до bf16 (у референса bf16, у нас fp16 -- разная мантисса!)
    ref_bf16_bits = w_ref.view(torch.int16).numpy()
    got_bf16 = mx.array(got32).astype(mx.bfloat16)
    got_bf16_bits = np.array(got_bf16.view(mx.uint16)).astype(np.int16)
    n_mis = int((ref_bf16_bits != got_bf16_bits).sum())

    N = 20
    t0 = time.time()
    for _ in range(N):
        w = gw._dequant_w()
        mx.eval(w)
    dt = (time.time() - t0) / N * 1000
    total += dt if generic.startswith("blocks") else 0.0
    print(f"{generic:30s} {qt.bits:<4d} {str(tuple(qt.shape)):18s} {n_mis:6d}/{w_ref.numel():<8d} {diff.max():13.6g} {dt:8.3f}")

print(f"\nsum sb6-only per layer (proj+cmix) = {total:.2f} ms -> x24 = {total*24:.1f} ms full forward")
