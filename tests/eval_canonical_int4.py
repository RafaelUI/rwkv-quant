"""Canonical (naive) per-row RTN INT4 -- replaces canonical INT6 as the
naive baseline (19.07-10 follow-up, А.'s point: canonical INT6 has no
real sub-byte packing in this codebase, so it lands at the SAME size as
canonical INT8 -- biggest file of the whole comparison, which reads as a
strawman even though it's an honest artifact of the naive scheme. INT4
gets real nibble packing (writer.pack_int4), so it actually lands in a
comparable size range to COMPRESSION/REDUCTION -- a fairer naive baseline.

Same shape as eval_canonical_int6.py: plain per-row RTN, no groupwise
scale, no AW weighting, no outlier handling -- what a generic INT4
quantizer would produce. Expect this to be much worse on ppl than
COMPRESSION (which is also int4-heavy but groupwise+AW+outlier-aware) --
that gap IS the point of the comparison."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.formats.schema import QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
OUT_PATH = "/tmp/canonical_int4.rwkvq"

CANONICAL_INT4 = QuantConfig(
    proj=4, cmix=4, emb_head=4,
    w_lora=4, a_lora=4, v_lora=4, g_lora=4,
    small=8,  # matches every other preset in this project (small tensors
              # never go below 8 bits anywhere -- COMPRESSION/REDUCTION/
              # mlx_int6_emu all use small=8 or 16; also sidesteps a
              # pack_int4 limitation on the 3D (1,1,C) small tensors that
              # only bites at bits<=4, unrelated to the comparison itself
              # since small is ~150KB total regardless of bit width)
    outlier_fracs={},
)

sd = torch.load(CKPT, map_location="cpu")
t0 = time.time()
tensors = {k: quantize_tensor(k, w, CANONICAL_INT4) for k, w in sd.items()}
del sd
ckpt = QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                            head_size=64, vocab_size=65536,
                            tensors=tensors, config_repr=repr(CANONICAL_INT4))
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
print(f"canonical_int4 KERNEL   ppl={ppl_kernel:14.4f}")

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
