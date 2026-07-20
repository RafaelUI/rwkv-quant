import sys, time, re
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests/dev")
import mlx.core as mx
from rwkv_quant.formats import reader
from mlx_dequant_proto import dequantize_gw_sb6_mlx, dequantize_gw_asym_mlx

ckpt = reader.load_raw("/tmp/reduction_v2.rwkvq")

# найдём один экземпляр каждой уникальной "формы" тензора среди blocks.0.*
targets = {}
for k, qt in ckpt.tensors.items():
    if not k.startswith("blocks.0."):
        continue
    if qt.gw_mode not in ("sb6", "asym"):
        continue
    generic = re.sub(r"blocks\.\d+\.", "blocks.N.", k)
    targets[generic] = (k, qt)

print(f"{len(targets)} уникальных форм на слой\n")

# warm-up компиляции кернелей на каждой форме
fns = {"sb6": dequantize_gw_sb6_mlx, "asym": dequantize_gw_asym_mlx}
for generic, (k, qt) in targets.items():
    fn = fns[qt.gw_mode]
    w = fn(qt); mx.eval(w)

total_per_layer = 0.0
print(f"{'tensor':35s} {'mode':5s} {'bits':4s} {'shape':18s} {'warm_ms':8s}")
for generic, (k, qt) in sorted(targets.items()):
    fn = fns[qt.gw_mode]
    N = 20
    t0 = time.time()
    for _ in range(N):
        w = fn(qt)
        mx.eval(w)
    dt = (time.time() - t0) / N * 1000
    total_per_layer += dt
    print(f"{generic:35s} {qt.gw_mode:5s} {qt.bits:<4d} {str(tuple(qt.shape)):18s} {dt:8.2f}")

print(f"\nsum per layer (all unique tensors once) = {total_per_layer:.2f} ms")
print(f"x 24 layers ~= {total_per_layer*24:.1f} ms per full-model forward pass (proj+cmix only)")

# и отдельно emb+head
for k in ("emb.weight", "head.weight"):
    if k in ckpt.tensors:
        qt = ckpt.tensors[k]
        fn = fns[qt.gw_mode]
        w = fn(qt); mx.eval(w)
        t0 = time.time()
        for _ in range(5):
            w = fn(qt); mx.eval(w)
        dt = (time.time()-t0)/5*1000
        print(f"{k:20s} {qt.gw_mode:5s} bits={qt.bits} shape={tuple(qt.shape)} warm_ms={dt:.1f}")
