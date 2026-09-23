"""Первый настоящий прогон библиотечного measure на Mac (22.09 ночь).
    python mac_measure_2209.py <ckpt> <метка>  -> ~/Develop/WKV-kvant/sym_2209/libmac_<метка>_tau<t>.json"""
import copy, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats, autopick as ap
CK, TAG = sys.argv[1], sys.argv[2]
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
O = os.path.expanduser("~/Develop/WKV-kvant/sym_2209")
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split("M")[0])
if os.environ.get("RWKVQ_QUOTA"):   # диагностика: "en:3,ru:3,sr:2"
    ap.QUOTA = tuple((l, int(n)) for l, n in (x.split(":") for x in os.environ["RWKVQ_QUOTA"].split(",")))
    print("КВОТА", ap.QUOTA, flush=True)
s0 = sw(); t0 = time.time()
_, sig = act_stats.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(act_stats.CACHE_DIR, "act_%s.pt" % sig)
print("ACT", cfg.act_stats_path, flush=True)
m = ap.measure(CK, cfg, TOK, device="mps")
print("measure %.0f с (всего %.0f с), своп %+.0f МБ, окна %s" % (m["seconds"], time.time() - t0, sw() - s0, m["langs"]), flush=True)
for t in (5, 6, 7):
    ovr, rep = ap.select(m, t)
    json.dump(ovr, open("%s/libmac_%s_tau%d.json" % (O, TAG, t), "w"), indent=1)
    print("tau %d: вверх %d, вниз %d, байты %+.1f МБ, предск. KL %+.1f%%" % (t, rep["n_up"], rep["n_down"], rep["bytes"] / 1e6, 100 * rep["kl_pred"]), flush=True)
