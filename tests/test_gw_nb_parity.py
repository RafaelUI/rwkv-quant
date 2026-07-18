"""Гейт N-батчевого кернеля (_get_kernel_gw_nb): бит-в-бит с N=1-кернелем
(поколоночный прогон). Реальные тензоры: tmix int5, cmix int4, head int5
(чемпион) + proj int6 (reduction_v2). N = 2..12."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx, torch
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
lins = {
    "tmix r_proj int5 2048x2048": model.blocks[0].tmix.r_proj,
    "cmix key   int4 8192x2048": model.blocks[0].cmix.key,
    "head       int5 65536x2048": model.head,
}
red = load_raw("/tmp/reduction_v2.rwkvq")
k6 = next(k for k, qt in red.tensors.items()
          if getattr(qt, "gw_mode", "") == "sb6" and qt.gw_qh2 is not None)
lins[f"int6 {k6}"] = GwQuantLinear(red.tensors[k6])
del red

rng = np.random.default_rng(1)
fails = 0
for name, l in lins.items():
    for N in (2, 3, 4, 5, 8, 12):
        x = mx.array(rng.standard_normal((N, l.in_features)).astype(np.float32))
        mx.eval(x)
        ref = mx.concatenate([l(x[i:i+1]) for i in range(N)], axis=0)
        got = l(x)
        mx.eval(ref, got)
        r, g = np.array(ref), np.array(got)
        exact = np.array_equal(r, g)
        md = float(np.max(np.abs(r - g)))
        status = "BITEXACT" if exact else f"maxdiff={md:.3e}"
        if not exact and md > 0: fails += (md > 1e-5)
        print(f"{name:28s} N={N:2d}: {status}")
print("FAIL" if fails else "OK: все совпадения бит-в-бит либо в нуле")
