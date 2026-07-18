"""Canonical (naive) per-row RTN INT6 -- baseline for the MollySophia/
COMPRESSION/REDUCTION comparison requested 19.07-10. Plain asymmetric-free
RTN per output row (writer._real_quantize), NO groupwise scale, NO AW
weighting, NO outlier handling -- i.e. what a generic/off-the-shelf int6
quantizer (bitsandbytes-style) would produce. Same script shape as
eval_reduction_v2_real.py: real quantize+save -> size -> ppl (kernel path,
QuantLinearV2/v1 since no group_scale is set) -> decode speed.

NOTE (see writer._make_qt): bits>4 without group_scale stores codes as
raw int8 (no sub-byte packing exists for 5/6/7/8-bit in this codebase's
v1 real backend) -- so canonical INT6 is expected to occupy the SAME disk
size / decode bandwidth as canonical INT8 per-row. That is itself the
point of the comparison: naive bit-width choice alone does not buy real
compression without a dedicated packer, unlike this project's gw sb6
scheme (COMPRESSION/REDUCTION) or MLX's native affine 6-bit packing
(MollySophia)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import load_raw, _dequantize_one
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
OUT_PATH = "/tmp/canonical_int6.rwkvq"

CANONICAL_INT6 = QuantConfig(
    proj=6, cmix=6, emb_head=6,
    w_lora=6, a_lora=6, v_lora=6, g_lora=6, small=6,
    outlier_fracs={},
)

sd = torch.load(CKPT, map_location="cpu")
t0 = time.time()
tensors = {k: quantize_tensor(k, w, CANONICAL_INT6) for k, w in sd.items()}
del sd
ckpt = QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                            head_size=64, vocab_size=65536,
                            tensors=tensors, config_repr=repr(CANONICAL_INT6))
torch.save(ckpt, OUT_PATH)
print(f"quantize+save: {time.time()-t0:.1f}s, "
      f"file: {os.path.getsize(OUT_PATH)/1e6:.1f} MB")
del ckpt, tensors

data = torch.load(CORPUS)[:8].numpy()

def ppl_of(model):
    total_nll, total_tok = 0.0, 0
    for i in range(0, data.shape[0], 4):
        batch = data[i:i+4]
        idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
        logits = model(idx); mx.eval(logits)
        logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-12))
        B, T, V = logp.shape
        idxf = target.reshape(-1); logpf = logp.reshape(-1, V)
        nll = -logpf[np.arange(len(idxf)), idxf]
        total_nll += nll.sum(); total_tok += nll.size
    return float(np.exp(total_nll/total_tok))

ckpt2 = load_raw(OUT_PATH)
model = QuantRWKV7(ckpt2)
ppl_kernel = ppl_of(model)
print(f"canonical_int6 KERNEL   ppl={ppl_kernel:14.4f}")

prompt = mx.array(data[0:1, :64].astype(np.int32))
st = model.init_state(1)
logits, st = model.forward_stateful(prompt, st, last_only=True)
tok = mx.argmax(logits[:, -1], axis=-1)
for _ in range(8):
    logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
t0 = time.time(); n = 64
for _ in range(n):
    logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
dt = (time.time()-t0)/n*1000
print(f"decode: {dt:.2f} ms/tok")
