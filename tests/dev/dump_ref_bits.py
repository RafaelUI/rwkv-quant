import sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import mlx.core as mx
from rwkv_quant.formats import reader

ckpt = reader.load_raw("/tmp/reduction_v2.rwkvq")
keys = ["blocks.0.att.key.weight", "blocks.5.ffn.value.weight", "emb.weight"]
out = {}
for k in keys:
    w = reader._dequantize_gw_sb6(ckpt.tensors[k])  # torch bf16
    out[k.replace(".", "_")] = mx.array(w.view(__import__("torch").uint16).numpy()).view(mx.bfloat16)
mx.save_safetensors("/tmp/ref_bits_check.safetensors", out)
print("dumped", list(out.keys()))
