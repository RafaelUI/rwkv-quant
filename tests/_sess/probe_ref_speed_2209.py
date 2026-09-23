"""Цена одного прохода эталона RWKV7Ref на Mac (22.09 ночь): 8 окон x 511, fp32
счёт. Нужна для оценки measure на Mac. Печатает время двух проходов и своп."""
import os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK = os.path.expanduser(sys.argv[1]); DEV = sys.argv[2] if len(sys.argv) > 2 else "mps"
B = int(os.environ.get("B", "8"))
sw = lambda: subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split()[0]
s0 = sw(); t0 = time.time()
M = RWKV7Ref(CK, device=DEV, dtype=torch.bfloat16, compute_dtype=torch.float32)
blob = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
data = blob["tokens"][:B, :512][:, :-1].contiguous().to(DEV)
print("загрузка %.1f с" % (time.time() - t0), flush=True)
for i in range(2):
    t1 = time.time()
    with torch.no_grad():
        lg = M.forward(data, return_hidden=True)
    if DEV == "mps": torch.mps.synchronize()
    print("проход %d: %.1f с  (B=%d, T=%d)" % (i, time.time() - t1, data.shape[0], data.shape[1]), flush=True)
print("своп до %s после %s" % (s0, sw()))
