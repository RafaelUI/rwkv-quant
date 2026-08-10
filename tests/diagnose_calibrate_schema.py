"""Диагностика: КРИТЕРИЙ ПОИСКА calibrate() НЕ ЕСТЬ ПРОДАКШН-СХЕМА.

Что проверяется. `api.calibrate()` меряет ppl через `ablation.perplexity`
-> `RWKV7Ref.forward(cfg)` -> `calibration.fake_quant.q()`. Эта функция
знает ровно три вещи: симметричный per-row RTN, percentile-clip и
SpQR-надстройку. Про `cfg.group_scale` / `cfg.group_scale_mode` она не
знает НИЧЕГО (см. докстринг QuantConfig.group_scale: "Применяется только
в writer.quantize_tensor"). Пресеты же собраны ровно на group_scale.

Отсюда два независимых следствия, и скрипт меряет оба:

  A. КРИТЕРИЙ. calibrate() выбирает битность по деградации per-row
     схемы. Если per-row и groupwise ранжируют группы по-разному, выбор
     битности сделан не по той функции, которую потом деплоят.

  B. АРТЕФАКТ. Возвращённый QuantConfig не содержит group_scale, поэтому
     quantize(config=calibrated) уходит в per-row ветку quantize_tensor
     и производит файл, качество которого равно колонке "per-row", а не
     "groupwise". Разница и есть цена дефекта.

Запуск (0.1B, быстро):
    python tests/diagnose_calibrate_schema.py
Переменные окружения:
    RWKVQ_CKPT   путь к .pth      (умолчание -- 0.1B драфт)
    RWKVQ_NSEQ   число последовательностей корпуса (умолчание 6)
    RWKVQ_SEQLEN длина среза      (умолчание 256)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 6))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 256))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))

# Битности, на которых сравниваем две схемы. Берём сетку calibrate()
# пересечённую с тем, что умеет sb6 (4/5/6).
GRID = {
    "proj": (4, 5, 6),
    "cmix": (4, 5, 6),
    "emb_head": (4, 5, 6),
}
# Блоки как в пресетах.
GS = {"proj": 32, "cmix": 32, "emb_head": 32}


def load_corpus():
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    return tok[:NSEQ, :SEQLEN].contiguous()


def cfg_perrow(group, bits):
    """Ровно то, что строит calibrate(): битность группы и больше ничего."""
    return QuantConfig(**{group: bits})


def cfg_groupwise(group, bits, mode):
    """Продакшн-схема: та же битность, но блочный asym-scale."""
    return QuantConfig(**{group: bits},
                       group_scale={group: GS[group]},
                       group_scale_mode={group: mode})


def main():
    data = load_corpus()
    print(f"чекпоинт: {CKPT}")
    print(f"корпус:   {data.shape[0]}x{data.shape[1]} = "
          f"{data.shape[0] * (data.shape[1] - 1)} предсказаний\n")

    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    data = data.to("mps")

    t0 = time.time()
    base = perplexity(model, data, QuantConfig())
    print(f"BASELINE bf16  ppl={base:.4f}   [{time.time()-t0:.1f}s/прогон]\n")

    print(f"{'группа':10s} {'бит':>4s} | {'per-row (что меряет calibrate)':>30s} "
          f"| {'groupwise sb6 (что деплоят)':>28s} | {'ранг':>6s}")
    print("-" * 96)

    rows = []
    for group, bits_grid in GRID.items():
        for bits in bits_grid:
            p1 = perplexity(model, data, cfg_perrow(group, bits))
            d1 = 100 * (p1 - base) / base
            p2 = perplexity(model, data, cfg_groupwise(group, bits, "asym_sb6"))
            d2 = 100 * (p2 - base) / base
            rows.append((group, bits, d1, d2))
            print(f"{group:10s} {bits:4d} | {p1:14.4f}  {d1:+12.2f}% "
                  f"| {p2:12.4f}  {d2:+11.2f}% | {d1/d2 if d2 else float('nan'):6.1f}x")

    print("\nЧто отсюда следует:")
    print("  - колонка per-row -- функция, по которой calibrate() выбирает битность;")
    print("  - колонка groupwise -- функция, которая реально попадает в файл;")
    print("  - если множители не константа, то это НЕ смещение шкалы, а другой")
    print("    порядок предпочтений, и выбор битности сделан не по той функции.")

    import json
    out = os.environ.get("RWKVQ_OUT", "/tmp/calibrate_schema_gap.json")
    with open(out, "w") as f:
        json.dump({"ckpt": CKPT, "baseline": base,
                   "n_pred": int(data.shape[0] * (data.shape[1] - 1)),
                   "rows": rows}, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
