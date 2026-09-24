"""measure v3 (KL по окнам) с ШИРОКОЙ квотой en4/code4/ru2/sr2/zh2 на Mac (24.09): из одного
измерения reweight даёт выбор под любые веса языков без переизмерения.
    python mac_measure_wide_2409.py <ckpt>"""
import copy, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap
CK = sys.argv[1]
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split("M")[0])
ap.QUOTA = (("en", 4), ("code", 4), ("ru", 2), ("sr", 2), ("zh", 2))
s0 = sw()
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
m = ap.measure(CK, cfg, TOK, device="mps")
print("подпись %s, окна %s, %d матриц, KL %.6f, %.0f с, своп %+.0f МБ" % (m["signature"], m["langs"], len(m["arms"]), m["kl_all"], m["seconds"], sw() - s0), flush=True)
