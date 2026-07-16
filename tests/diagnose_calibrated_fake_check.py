"""Fake_quant-оценка итогового calibrated_e2e-конфига на ТОМ ЖЕ срезе
корпуса [:8], что и real-quant результат (16.9027), для честного
сопоставления яблок с яблоками (calibrate() сам считал на полных 24 chunks)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")

cfg = QuantConfig(
    proj=4, cmix=4, emb_head=4,
    w_lora=4, a_lora=4, v_lora=4, g_lora=6,
    small=6,
    outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    clip_percentiles={"small": 99.9},
)

model = RWKV7Ref(CKPT_PTH, device="cpu", dtype=torch.bfloat16)
data = torch.load(CORPUS)[:8]
baseline = perplexity(model, data, QuantConfig(), batch_size=4)
ppl = perplexity(model, data, cfg, batch_size=4)
print(f"baseline_fake ppl={baseline:.4f}")
print(f"calibrated_e2e_fake ppl={ppl:.4f}  delta={100*(ppl-baseline)/baseline:+.2f}%")
