"""E2E A/B полублока (K3_HALF on/off) на ОДНОМ экземпляре модели,
чередование раундов. Аргумент -- путь к .rwkvq."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

PRESET = sys.argv[1] if len(sys.argv) > 1 else "/tmp/champion_v2.rwkvq"
m = QuantRWKV7(load_raw(PRESET))
rng = np.random.default_rng(0)

# санити
idx = mx.array(rng.integers(0, 65000, (1, 4)).astype(np.int64))
st0 = m.init_state(1)
gw.K3_HALF = False; lg0, _ = m.forward_stateful(idx, st0); mx.eval(lg0)
gw.K3_HALF = True;  lg1, _ = m.forward_stateful(idx, st0); mx.eval(lg1)
a, b = np.array(lg0), np.array(lg1)
print(f"sanity: maxabs {float(np.abs(a-b).max()):.4f} "
      f"top1 same: {bool((a.argmax(-1)==b.argmax(-1)).all())}", flush=True)

def dec_round(st, n=32):
    tok = mx.array(rng.integers(0, 65000, (1, 1)).astype(np.int64)); ts = []
    for _ in range(4):
        lg, st = m.step(tok, st); mx.eval(lg)
    mx.synchronize()
    for _ in range(n):
        t0 = time.perf_counter(); lg, st = m.step(tok, st); mx.eval(lg); mx.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return ts, st

warm = mx.array(rng.integers(0, 65000, (1, 64)).astype(np.int64))
st = m.init_state(1); lg, st = m.forward_stateful(warm, st, last_only=True); mx.eval(lg)
res = {True: [], False: []}
for r in range(5):
    for flag in (True, False):
        gw.K3_HALF = flag
        a, st = dec_round(st); res[flag] += a
mon, moff = np.median(res[True]), np.median(res[False])
print(f"{os.path.basename(PRESET)} decode: HALF={mon:.2f}  full={moff:.2f} ms/tok  x{moff/mon:.3f}")
