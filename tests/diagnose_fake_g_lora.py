"""fake_quant-эквивалент bisection: g_lora=4 alone через RWKV7Ref (torch,
device=cpu), тот же корпус/срез, что и diagnose_one.py, для прямого
сравнения с real-quant результатом (g_lora_only real ppl=167.2455)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")

model = RWKV7Ref(CKPT_PTH, device="cpu", dtype=torch.bfloat16)
data = torch.load(CORPUS)[:8]

for name, cfg in [
    ("baseline_fp", QuantConfig()),
    ("g_lora_only_fake", QuantConfig(g_lora=4)),
    ("w_lora_only_fake", QuantConfig(w_lora=4)),
    ("a_lora_only_fake", QuantConfig(a_lora=4)),
    ("v_lora_only_fake", QuantConfig(v_lora=4)),
]:
    ppl = perplexity(model, data, cfg, batch_size=4)
    print(f"{name:20s} ppl={ppl:14.4f}")
