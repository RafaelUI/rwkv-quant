"""Сквозная валидация REDUCTION v2 (int6, xbits=2): реальная упаковка ->
.rwkvq на диск (размер) -> load -> ppl двумя путями (dense-dequant
референс И кернельный GwQuantLinear-путь, крест-проверка друг с другом
и с фейковым числом 11.4426 из diagnose_one) -> decode-скорость кернеля
(первый замер int6, сравнить с чемпионом int4/int5 ~15.5 мс/ток)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import load_raw, _dequantize_one
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.presets import REDUCTION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
OUT_PATH = "/tmp/reduction_v2.rwkvq"

sd = torch.load(CKPT, map_location="cpu")
t0 = time.time()
tensors = {k: quantize_tensor(k, w, REDUCTION, real_gw=True) for k, w in sd.items()}
del sd
ckpt = QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                            head_size=64, vocab_size=65536,
                            tensors=tensors, config_repr=repr(REDUCTION))
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

# --- путь 1: dense-dequant референс (та же математика, что diagnose_one) ---
ckpt = load_raw(OUT_PATH)
dense = {k: QuantizedTensor(key=k, group=qt.group, bits=16, shape=qt.shape,
                             dense=_dequantize_one(qt))
         for k, qt in ckpt.tensors.items()}
ckpt.tensors = dense
ref_model = QuantRWKV7(ckpt)
ppl_dense = ppl_of(ref_model)
print(f"reduction_v2_real DENSE-DEQUANT   ppl={ppl_dense:14.4f}  (fake-путь был 11.4426)")

# --- путь 2: кернельный (GwQuantLinear, без разворачивания в dense) ---
ckpt2 = load_raw(OUT_PATH)
model = QuantRWKV7(ckpt2)
ppl_kernel = ppl_of(model)
print(f"reduction_v2_real KERNEL          ppl={ppl_kernel:14.4f}")

# --- decode-скорость кернельного пути ---
prompt = mx.array(data[0:1, :64].astype(np.int32))
st = model.init_state(1)
logits, st = model.forward_stateful(prompt, st, last_only=True)
tok = mx.argmax(logits[:, -1], axis=-1)
for _ in range(8):   # прогрев compile
    logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
t0 = time.time(); n = 64
for _ in range(n):
    logits, st = model.step(tok[None], st); tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
dt = (time.time()-t0)/n*1000
print(f"decode: {dt:.2f} ms/tok  (чемпион int4/int5 fused ~15.5 мс/ток для сравнения)")
