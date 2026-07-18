"""Та же кривая T=1..32, но модель affine6 (mx.quantized_matmul)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor, _match_group
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.presets import REDUCTION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
GS = 64; BITS = {"proj": 6, "cmix": 6, "emb_head": 6}

def make_affine_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits); mx.eval(wq, s, b)
    qt = QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(w.shape))
    qt.gw_mode = "mlx_affine"; qt.mlx_weight = wq; qt.mlx_scales = s
    qt.mlx_biases = b; qt.mlx_group_size = GS; qt.mlx_bits = bits
    return qt

def emb_roundtrip_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits)
    deq = mx.dequantize(wq, scales=s, biases=b, group_size=GS, bits=bits); mx.eval(deq)
    dense = torch.from_numpy(np.array(deq.astype(mx.float32))).to(torch.bfloat16)
    return QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape), dense=dense)

sd = torch.load(CKPT, map_location="cpu")
tensors = {}
for k in list(sd.keys()):
    w = sd.pop(k); g = _match_group(k)
    if g in BITS and w.dim() == 2 and w.shape[1] % GS == 0:
        tensors[k] = emb_roundtrip_qt(k, g, w, BITS[g]) if "emb" in k \
                     else make_affine_qt(k, g, w, BITS[g])
    else:
        tensors[k] = quantize_tensor(k, w, REDUCTION, real_gw=True)
del sd
model = QuantRWKV7(QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                   head_size=64, vocab_size=65536, tensors=tensors, config_repr="affine"))
rng = np.random.default_rng(0)

def bench_T(T, reps=20, warm=5):
    idx = mx.array(rng.integers(0, 65000, (1, T)).astype(np.int64))
    warm_idx = mx.array(rng.integers(0, 65000, (1, 64)).astype(np.int64))
    st = model.init_state(1)
    lg, st = model.forward_stateful(warm_idx, st, last_only=True); mx.eval(lg)
    for _ in range(warm):
        lg, _ = model.forward_stateful(idx, st, last_only=False); mx.eval(lg)
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lg, _ = model.forward_stateful(idx, st, last_only=False); mx.eval(lg)
        mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return float(np.median(ts))

print(f"{'T':>3s} {'ms/проход':>10s} {'ms/ток':>8s} {'vs T=1':>7s}")
base = None
for T in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    ms = bench_T(T)
    if base is None: base = ms
    print(f"{T:3d} {ms:10.2f} {ms/T:8.2f} {ms/base:6.2f}x", flush=True)
