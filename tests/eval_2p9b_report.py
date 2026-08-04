"""Сводная таблица по результатам tests/eval_2p9b_one.py (по конфигу на
процесс -- см. там про своп). Печатает Δ% к bf16 по языкам."""
import json
import os

OUT_JSON = os.path.expanduser("~/Develop/WKV-kvant/eval_2p9b.json")
ORDER = ["bf16", "reduction", "reduction_small", "reduction_fix", "compression", "compression_fix"]

r = json.load(open(OUT_JSON))
base = r["bf16"]["ppl"]
langs = [k for k in base]
print(f"{'конфиг':<20}{'MB':>9}" + "".join(f"{l:>12}" for l in langs))
for name in ORDER:
    if name not in r:
        continue
    row = "".join(f"{100*(r[name]['ppl'][l]-base[l])/base[l]:>+11.2f}%"
                  for l in langs)
    print(f"{name:<20}{r[name]['size_mb']:>9.1f}{row}")
print("\nабсолютные ppl:")
for name in ORDER:
    if name in r:
        print(f"  {name:<20}" + "  ".join(
            f"{l}={r[name]['ppl'][l]:.4f}" for l in langs))
