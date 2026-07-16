"""Изоляция: что именно взрывает COMPRESSION -- proj+cmix+SpQR сами по себе,
или их комбинация с emb_head/small/lora?"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

sd = torch.load(CKPT_PTH, map_location="cpu")
data = torch.load(CORPUS)[:8].numpy()

def build(cfg):
    tensors = {key: quantize_tensor(key, w, cfg) for key, w in sd.items()}
    return QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                                head_size=HEAD_SIZE, vocab_size=VOCAB,
                                tensors=tensors, config_repr=repr(cfg))

@torch.no_grad()
def ppl(model, data_np, batch_size=4):
    total_nll, total_tok = 0.0, 0
    for i in range(0, data_np.shape[0], batch_size):
        batch = data_np[i:i+batch_size]
        idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
        logits = model(idx); mx.eval(logits)
        logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1)+1e-12))
        B,T,V = logp.shape
        idxf = target.reshape(-1); logpf = logp.reshape(-1,V)
        nll = -logpf[np.arange(len(idxf)), idxf]
        total_nll += nll.sum(); total_tok += nll.size
    return float(np.exp(total_nll/total_tok))

cases = [
    ("only proj INT4+SpQR2%", QuantConfig(proj=4, outlier_fracs={"proj":0.02})),
    ("only cmix INT4+SpQR2%", QuantConfig(cmix=4, outlier_fracs={"cmix":0.02})),
    ("only emb_head INT4+SpQR2%", QuantConfig(emb_head=4, outlier_fracs={"emb_head":0.02})),
    ("only small INT6 (no clip, писатель игнорирует clip)", QuantConfig(small=6)),
    ("only lora INT4 (w/a/v/g)", QuantConfig(w_lora=4,a_lora=4,v_lora=4,g_lora=4)),
    ("proj+cmix INT4+SpQR2% (без emb_head/lora/small)", QuantConfig(proj=4,cmix=4, outlier_fracs={"proj":0.02,"cmix":0.02})),
    ("proj+cmix+emb_head INT4+SpQR2% (без lora/small)", QuantConfig(proj=4,cmix=4,emb_head=4, outlier_fracs={"proj":0.02,"cmix":0.02,"emb_head":0.02})),
]

for name, cfg in cases:
    t0=time.time()
    ckpt = build(cfg)
    model = QuantRWKV7(ckpt)
    p = ppl(model, data)
    print(f"{name:55s} ppl={p:12.4f}  [{time.time()-t0:.1f}s]")
    del model, ckpt
