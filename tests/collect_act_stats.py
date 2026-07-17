"""Сбор E[x^2] по входным каналам квантуемых матриц (bf16 RWKV7Ref, dense).
Калибровочный срез корпуса [8:16] -- НЕ пересекается с измерительным [:8].
Выход: /tmp/act_stats_1p5b.pt = {state_dict_key: tensor[in_features] E[x^2]}."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import rwkv_quant.models.rwkv7_ref as ref_mod
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")

model = RWKV7Ref(CKPT_PTH, device="cpu", dtype=torch.bfloat16)
data = torch.load(CORPUS)[8:16]
ref_mod.ACT_RECORDER = {}
t0 = time.time()
with torch.no_grad():
    for i in range(data.shape[0]):
        model.forward(data[i:i+1, :-1])
        print(f"chunk {i+1}/{data.shape[0]} {time.time()-t0:.0f}s", flush=True)
stats = {k: (ss / max(n, 1)) for k, (ss, n) in ref_mod.ACT_RECORDER.items()}
torch.save(stats, "/tmp/act_stats_1p5b.pt")
print(f"saved {len(stats)} tensors -> /tmp/act_stats_1p5b.pt")
