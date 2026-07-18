"""Сквозная валидация формата v2: реальная упаковка чемпион-конфига ->
.rwkvq на диск (размер!) -> load -> деквант -> ppl тем же пайплайном,
что diagnose_one (QuantRWKV7 на dense bf16). Ожидание: ~11.668
(l6_proj5_emb5; допустим микросдвиг от half-роундтрипа d/dm)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import load_raw, _dequantize_one
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from test_v2_format import CHAMPION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
OUT_PATH = "/tmp/champion_v2.rwkvq"

sd = torch.load(CKPT, map_location="cpu")
t0 = time.time()
tensors = {k: quantize_tensor(k, w, CHAMPION, real_gw=True) for k, w in sd.items()}
del sd
ckpt = QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                            head_size=64, vocab_size=65536,
                            tensors=tensors, config_repr=repr(CHAMPION))
torch.save(ckpt, OUT_PATH)
print(f"quantize+save: {time.time()-t0:.1f}s, "
      f"file: {os.path.getsize(OUT_PATH)/1e6:.1f} MB")
del ckpt, tensors

ckpt = load_raw(OUT_PATH)
dense = {k: QuantizedTensor(key=k, group=qt.group, bits=16, shape=qt.shape,
                             dense=_dequantize_one(qt))
         for k, qt in ckpt.tensors.items()}
ckpt.tensors = dense
model = QuantRWKV7(ckpt)

data = torch.load(CORPUS)[:8].numpy()
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
print(f"champion_v2_real    ppl={float(np.exp(total_nll/total_tok)):14.4f}")
