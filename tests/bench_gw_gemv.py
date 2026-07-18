"""Микробенч GEMV: GwQuantLinear (sb6, int4/int5) против QuantLinearV2
packed (v1 per-row int4) на боевых формах 1.5B. Медианы по чередующимся
раундам, N=1."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear
from rwkv_quant.backends.metal.quant_linear_v2 import QuantLinearV2
from rwkv_quant.calibration.group_config import QuantConfig
from test_v2_format import CHAMPION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
sd = torch.load(CKPT, map_location="cpu")

V1 = QuantConfig(proj=4, cmix=4, emb_head=4, w_lora=4, a_lora=4, v_lora=4,
                 g_lora=8, small=8, outlier_fracs={})

def bench(lin, IN, reps=60):
    x = mx.array(np.random.randn(1, IN).astype(np.float32))
    for _ in range(10): mx.eval(lin(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(lin(x)); ts.append(time.perf_counter()-t0)
    return np.median(ts)*1000

for key in ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight",
            "blocks.0.att.receptance.weight", "head.weight"]:
    w = sd[key]; OUT, IN = w.shape
    gw = GwQuantLinear(quantize_tensor(key, w, CHAMPION, real_gw=True))
    v1 = QuantLinearV2(quantize_tensor(key, w, V1))
    bits = 5 if gw.has_qh else 4
    mb_gw = (OUT*IN*(bits+0.5))/8/1e6; mb_v1 = OUT*IN*0.5/1e6
    a, b = bench(gw, IN), bench(v1, IN)
    print(f"{key:34s} {str((OUT,IN)):14s} gw(int{bits},{mb_gw:5.1f}MB)={a:7.3f}ms "
          f"[{mb_gw/a:5.1f}GB/s]  v1(int4,{mb_v1:5.1f}MB)={b:7.3f}ms [{mb_v1/b:5.1f}GB/s]")
