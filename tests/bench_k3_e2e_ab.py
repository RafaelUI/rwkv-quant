"""E2E A/B кернель-3 vs старый: два экземпляра модели (K3 on/off) в ОДНОМ
процессе, чередование раундов (закон 1). Decode T=1 compiled step + verify
T=4/T=8 через forward_stateful. Чемпион и REDUCTION v2."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

PRESET = sys.argv[1] if len(sys.argv) > 1 else "/tmp/champion_v2.rwkvq"
rng = np.random.default_rng(0)

raw = load_raw(PRESET)
gw.K3 = True
m_new = QuantRWKV7(raw)
gw.K3 = False
m_old = QuantRWKV7(raw)
gw.K3 = True
del raw

def dec_round(m, st, n=32):
    ts = []
    tok = mx.array(rng.integers(0, 65000, (1, 1)).astype(np.int64))
    for _ in range(4):
        lg, st = m.step(tok, st); mx.eval(lg)
    mx.synchronize()
    for _ in range(n):
        t0 = time.perf_counter()
        lg, st = m.step(tok, st); mx.eval(lg)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts, st

def ver_round(m, st, T, n=12):
    idx = mx.array(rng.integers(0, 65000, (1, T)).astype(np.int64))
    ts = []
    for _ in range(3):
        lg, _ = m.forward_stateful(idx, st); mx.eval(lg)
    mx.synchronize()
    for _ in range(n):
        t0 = time.perf_counter()
        lg, _ = m.forward_stateful(idx, st); mx.eval(lg)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts

warm = mx.array(rng.integers(0, 65000, (1, 64)).astype(np.int64))
st_n = m_new.init_state(1); lg, st_n = m_new.forward_stateful(warm, st_n, last_only=True); mx.eval(lg)
st_o = m_old.init_state(1); lg, st_o = m_old.forward_stateful(warm, st_o, last_only=True); mx.eval(lg)

res = {"new": [], "old": []}
for r in range(4):
    a, st_n = dec_round(m_new, st_n); res["new"] += a
    b, st_o = dec_round(m_old, st_o); res["old"] += b
print(f"{os.path.basename(PRESET)} decode T=1 compiled: "
      f"new={np.median(res['new']):.2f}  old={np.median(res['old']):.2f} ms/tok  "
      f"x{np.median(res['old'])/np.median(res['new']):.3f}", flush=True)

for T in (4, 8):
    rn, ro = [], []
    for r in range(3):
        rn += ver_round(m_new, st_n, T)
        ro += ver_round(m_old, st_o, T)
    mn, mo = np.median(rn), np.median(ro)
    print(f"verify T={T} raw: new={mn:.2f} ({mn/T:.2f}/ток x{mn/np.median(res['new'])/T*1:.2f} от T=1)  "
          f"old={mo:.2f}  выигрыш x{mo/mn:.3f}", flush=True)
