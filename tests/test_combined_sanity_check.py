"""
Проверка combined_sanity_check() на РЕАЛЬНОМ кейсе, который взорвал
COMPRESSION: все 4 LoRA-ветки на INT4 одновременно (ppl 11.4 -> 248 на
rwkv7-g1h-1.5b, см. diagnose_one.py). Ожидаем: guard детектирует взрыв
относительно допуска и откатывает g_lora (или другую branch) вверх по
битности, пока не уложится в допуск.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _ckpt import ckpt_1p5b  # 1.5B: RWKVQ_CKPT_1P5B, g1h или g1j
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.calibration.ablation import perplexity, combined_sanity_check

CKPT_PTH = ckpt_1p5b()
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
start_bits = dict(best_bits)

best_bits, outlier_fracs, final_ppl, final_delta = combined_sanity_check(
    model, data, best_bits, outlier_fracs, clip_percentiles,
    baseline_ppl=baseline_ppl, ppl_threshold_pct=5.0, verbose=True)

print(f"\nfinal best_bits = {best_bits}")
print(f"final ppl={final_ppl:.4f}  Δ={final_delta:+.2f}%")

# Гейт проверяет ОТКАТ. Если на этом чекпоинте все LoRA на INT4 и так в допуске
# (g1j 1.5B: +13 % при допуске 25 %), откатывать нечего, и «в допуске» -- не
# проверка guard'а: справочный код 2, как у гейтов без нужных данных.
if best_bits == start_bits:
    print(f"\n[НЕ ПРИМЕНИМ] взрыва нет (Δ={final_delta:+.2f}% < 25%), guard ничего не откатил -- "
          f"кейс воспроизводится на rwkv7-g1h-1.5b (RWKVQ_CKPT_1P5B)")
    sys.exit(2)
ok = final_delta < 25.0
print(f"\n[{'OK' if ok else 'FAIL'}] guard привёл конфиг в допуск: {ok}")
sys.exit(0 if ok else 1)
