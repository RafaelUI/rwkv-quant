"""Численная сверка GwQuantLinear (GEMV sb6 + GEMM-путь) с референсом
x @ dequant(qt).T в fp32 на живых тензорах чемпион-конфига."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import _dequantize_one
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear
from test_v2_format import CHAMPION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
sd = torch.load(CKPT, map_location="cpu")
torch.manual_seed(0)

for key in ["blocks.0.ffn.key.weight",        # int4, IN=2048, OUT=8192
            "blocks.0.ffn.value.weight",      # int4, IN=8192
            "blocks.0.att.receptance.weight", # int5 (qh), 2048x2048
            "head.weight"]:                   # int5, OUT=65536
    qt = quantize_tensor(key, sd[key], CHAMPION, real_gw=True)
    lin = GwQuantLinear(qt)
    ref_w = _dequantize_one(qt).float().numpy()
    for N in (1, 3, 16):
        x = torch.randn(N, qt.shape[1]).numpy().astype(np.float32)
        y = np.array(lin(mx.array(x)))
        ref = x @ ref_w.T
        rel = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9)
        path = "GEMM" if N >= 16 else "GEMV"
        print(f"{key:34s} N={N:2d} {path} relmax={rel:.3e}")
        assert rel < 3e-3, (key, N, rel)
print("KERNEL NUMERICS OK")
