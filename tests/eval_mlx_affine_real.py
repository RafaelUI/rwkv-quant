"""REDUCTION v3 кандидат: тело (proj/cmix/head) в НАТИВНОМ MLX affine
формате (mx.quantize gs=64) -> MlxAffineQuantLinear (mx.quantized_matmul,
тот же быстрый путь, что у MollySophia: bench_mlx_vs_gw_ab показал +37-42%
на рабочих формах GEMV и +26-49% на GEMM-префилле против gw-кернеля).
emb -- mx.quantize roundtrip -> dense fp16 (lookup, как у Molly).
LoRA/small -- как в presets.REDUCTION (real_gw, gw64@6 / int8).
ppl на eval_corpus_world.pt[:8] + decode-скорость. Один кейс = один
процесс (закон N2). Кейсы: p6c6e6 (REDUCTION v3), p5c4e5 (размер
COMPRESSION), p6c4e6.
Внимание: decode-число тут -- одиночный процесс; для отчётных цифр
прогнать A/B-чередованием с чемпионом (закон N1).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor, _match_group
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.presets import REDUCTION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
GS = 64

CASES = {
    "p6c6e6": {"proj": 6, "cmix": 6, "emb_head": 6},
    "p5c4e5": {"proj": 5, "cmix": 4, "emb_head": 5},
    "p6c4e6": {"proj": 6, "cmix": 4, "emb_head": 6},
}
name = sys.argv[1]
BITS = CASES[name]

def make_affine_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits)
    mx.eval(wq, s, b)
    qt = QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(w.shape))
    qt.gw_mode = "mlx_affine"
    qt.mlx_weight = wq; qt.mlx_scales = s; qt.mlx_biases = b
    qt.mlx_group_size = GS; qt.mlx_bits = bits
    qt._mb = (wq.size*4 + s.size*2 + b.size*2) / 1e6
    return qt

def emb_roundtrip_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits)
    deq = mx.dequantize(wq, scales=s, biases=b, group_size=GS, bits=bits)
    mx.eval(deq)
    dense = torch.from_numpy(np.array(deq.astype(mx.float32))).to(torch.bfloat16)
    qt = QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape), dense=dense)
    qt._mb = (wq.size*4 + s.size*2 + b.size*2) / 1e6  # честный учёт как квантованного
    return qt

sd = torch.load(CKPT, map_location="cpu")
t0 = time.time()
tensors, mb_body, mb_rest = {}, 0.0, 0.0
for k in list(sd.keys()):
    w = sd.pop(k)
    g = _match_group(k)
    if g in BITS and w.dim() == 2 and w.shape[1] % GS == 0:
        if "emb" in k:
            qt = emb_roundtrip_qt(k, g, w, BITS[g])
        else:
            qt = make_affine_qt(k, g, w, BITS[g])
        mb_body += qt._mb
        tensors[k] = qt
    else:
        qt = quantize_tensor(k, w, REDUCTION, real_gw=True)
        tensors[k] = qt
del sd
print(f"quantize: {time.time()-t0:.1f}s; тело(affine {BITS}) = {mb_body:.1f} MB", flush=True)

ckpt = QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                            head_size=64, vocab_size=65536,
                            tensors=tensors, config_repr=f"mlx_affine {BITS} + REDUCTION rest")
model = QuantRWKV7(ckpt)
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

ppl = ppl_of(model)
print(f"mlx_affine_{name}  ppl={ppl:14.4f}  (bf16 11.430, COMPRESSION 11.7125, REDUCTION v2 11.4438)", flush=True)

# decode (кернельный путь mx.quantized_matmul)
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
print(f"decode: {dt:.2f} ms/tok (одиночный процесс, для отчёта нужен A/B)", flush=True)

# prefill T=1024
xp = mx.array(data[0:1, :1024].astype(np.int32))
st2 = model.init_state(1)
for _ in range(2):
    lg, _ = model.forward_stateful(xp, model.init_state(1), last_only=True); mx.eval(lg)
t0 = time.time()
for _ in range(3):
    lg, _ = model.forward_stateful(xp, model.init_state(1), last_only=True); mx.eval(lg)
pt = (time.time()-t0)/3
print(f"prefill T=1024: {1024/pt:.0f} tok/s (одиночный процесс)", flush=True)
