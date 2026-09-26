"""measure v3 (широкая квота по умолчанию) на сервере: python srv_measure_wide_2409.py <ckpt> <tok> <device>"""
import copy, os, sys, time
sys.path.insert(0, os.path.expanduser("~/rwkvq/rwkv-quant-git"))
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap
CK, TOK, DEV = sys.argv[1:4]
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
print(ap.__file__); print("квота", ap.QUOTA, "act", sig, flush=True)
m = ap.measure(CK, cfg, TOK, device=DEV)
print("подпись %s, окна %s, %d матриц, KL %.6f, %.0f с" % (m["signature"], m["langs"], len(m["arms"]), m["kl_all"], m["seconds"]), flush=True)
