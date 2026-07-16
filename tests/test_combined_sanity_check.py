"""
Проверка combined_sanity_check() на РЕАЛЬНОМ кейсе, который взорвал
COMPRESSION: все 4 LoRA-ветки на INT4 одновременно (ppl 11.4 -> 248 на
rwkv7-g1h-1.5b, см. diagnose_one.py). Ожидаем: guard детектирует взрыв
относительно допуска и откатывает g_lora (или другую branch) вверх по
битности, пока не уложится в допуск.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.calibration.ablation import perplexity, combined_sanity_check

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")

model = RWKV7Ref(CKPT_PTH, device="cpu", dtype=torch.float32)
data = torch.load(CORPUS)[:8]

baseline_ppl = perplexity(model, data)
print(f"baseline ppl = {baseline_ppl:.4f}")

# намеренно воспроизводим сломанный кейс: все lora=4, ppl_threshold_pct=5%
# (значит explosion_threshold = 5*5=25% по умолчанию) -- взрыв в разы больше
best_bits = {"proj": 16, "cmix": 16, "emb_head": 16,
             "w_lora": 4, "a_lora": 4, "v_lora": 4, "g_lora": 4, "small": 16}
outlier_fracs, clip_percentiles = {}, {}

best_bits, outlier_fracs, final_ppl, final_delta = combined_sanity_check(
    model, data, best_bits, outlier_fracs, clip_percentiles,
    baseline_ppl=baseline_ppl, ppl_threshold_pct=5.0, verbose=True)

print(f"\nfinal best_bits = {best_bits}")
print(f"final ppl={final_ppl:.4f}  Δ={final_delta:+.2f}%")

ok = final_delta < 25.0
print(f"\n[{'OK' if ok else 'FAIL'}] guard привёл конфиг в допуск: {ok}")
sys.exit(0 if ok else 1)
