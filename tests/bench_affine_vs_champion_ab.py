"""A/B в одном процессе (закон N1): champion_v2 (gw sb6) vs mlx-affine
p6c6e6 (mx.quantized_matmul). Decode 64 ток x3 раунда чередованием +
prefill T=1024 x3 чередованием, mx.synchronize после каждого, чек-сумма
логитов (гарантия, что граф реально исполнен)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor, _match_group
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.presets import REDUCTION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
GS = 64
BITS = {"proj": 6, "cmix": 6, "emb_head": 6}

def make_affine_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits)
    mx.eval(wq, s, b)
    qt = QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(w.shape))
    qt.gw_mode = "mlx_affine"
    qt.mlx_weight = wq; qt.mlx_scales = s; qt.mlx_biases = b
    qt.mlx_group_size = GS; qt.mlx_bits = bits
    return qt

def emb_roundtrip_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits)
    deq = mx.dequantize(wq, scales=s, biases=b, group_size=GS, bits=bits)
    mx.eval(deq)
    dense = torch.from_numpy(np.array(deq.astype(mx.float32))).to(torch.bfloat16)
    return QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape), dense=dense)

print("build affine model...", flush=True)
sd = torch.load(CKPT, map_location="cpu")
tensors = {}
for k in list(sd.keys()):
    w = sd.pop(k)
    g = _match_group(k)
    if g in BITS and w.dim() == 2 and w.shape[1] % GS == 0:
        tensors[k] = emb_roundtrip_qt(k, g, w, BITS[g]) if "emb" in k \
                     else make_affine_qt(k, g, w, BITS[g])
    else:
        tensors[k] = quantize_tensor(k, w, REDUCTION, real_gw=True)
del sd
aff = QuantRWKV7(QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                 head_size=64, vocab_size=65536, tensors=tensors, config_repr="affine"))
print("load champion...", flush=True)
champ = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
data = torch.load(CORPUS)[:8].numpy()
models = {"champion(gw)": champ, "affine6(mlx)": aff}

def decode_ms(model, n=64):
    prompt = mx.array(data[0:1, :64].astype(np.int32))
    st = model.init_state(1)
    logits, st = model.forward_stateful(prompt, st, last_only=True)
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(8):
        logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    return (time.perf_counter()-t0)/n*1000

def prefill_ts(model):
    xp = mx.array(np.random.randint(0, 65000, (1, 1024)).astype(np.int64))
    lg, _ = model.forward_stateful(xp, model.init_state(1), last_only=True)
    mx.eval(lg); mx.synchronize()   # прогрев
    t0 = time.perf_counter()
    lg, _ = model.forward_stateful(xp, model.init_state(1), last_only=True)
    mx.eval(lg); mx.synchronize()
    dt = time.perf_counter()-t0
    cs = float(mx.sum(mx.abs(lg.astype(mx.float32))))
    return 1024/dt, cs

dec = {n: [] for n in models}; pre = {n: [] for n in models}
for r in range(3):
    for n, m in models.items():
        dec[n].append(decode_ms(m))
    for n, m in models.items():
        ts, cs = prefill_ts(m)
        pre[n].append(ts)
        if r == 0: print(f"  {n} prefill checksum={cs:.1f}", flush=True)
    print(f"round {r}: " + " | ".join(
        f"{n}: dec {dec[n][-1]:.2f}ms pre {pre[n][-1]:.0f}t/s" for n in models), flush=True)

print("\nМедианы (3 раунда, A/B):")
for n in models:
    print(f"{n}: decode {np.median(dec[n]):.2f} ms/tok, prefill {np.median(pre[n]):.0f} tok/s")
