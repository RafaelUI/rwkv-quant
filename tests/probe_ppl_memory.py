"""Откуда берётся память на композитном замере ppl.

Повод. Изолированные прогоны (квантована одна группа) свопа не двигали,
а композит (квантованы ВСЕ) вырастил его с 0.3 до 5.8 ГБ на 1.5B.
Первое объяснение -- кеш квантованных весов -- оказалось неполным:
после введения бюджета в 1536 МБ своп продолжил расти. Значит виноват не
только он, и гадать дальше бессмысленно.

Здесь три конфигурации меряются подряд В ОДНОМ ПРОЦЕССЕ с выборкой свопа
и `vm_stat` по ходу (закон 11: RSS для unified memory бесполезен):

  1. bf16                -- пол: только модель и логиты, ноль квантования;
  2. композит, кеш ВЫКЛ  -- добавляются транзиенты блочного поиска;
  3. композит, кеш ВКЛ   -- добавляется удержание квантованных копий.

Разности между строками и есть ответ, какая статья сколько стоит.

    python tests/probe_ppl_memory.py
"""
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.calibration import fake_quant
from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.presets import PRESETS

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 6))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                       capture_output=True, text=True).stdout
    try:
        return float(s.split("used = ")[1].split("M")[0])
    except Exception:                                        # noqa: BLE001
        return -1.0


def free_pct():
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                         text=True).stdout
    try:
        return int(out.strip().split(":")[-1].strip().rstrip("%"))
    except Exception:                                        # noqa: BLE001
        return -1


class Watch:
    """Сэмплер свопа в фоне -- пик важнее конечного значения: своп не
    отдаётся системе сразу, и замер «после» занижает реальный расход."""

    def __init__(self, period=2.0):
        self.period, self.stop = period, False
        self.peak_swap, self.min_free = swap_mb(), free_pct()

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self.stop:
            self.peak_swap = max(self.peak_swap, swap_mb())
            f = free_pct()
            if f >= 0:
                self.min_free = min(self.min_free, f)
            time.sleep(self.period)

    def __exit__(self, *a):
        self.stop = True
        self.t.join(timeout=3)


def main():
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")
    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    red = PRESETS["reduction"]
    comp = QuantConfig(group_scale=dict(red.group_scale),
                       group_scale_mode=dict(red.group_scale_mode),
                       act_stats_path=None, **dict(red.bits))

    print(f"корпус {tuple(data.shape)}, старт: своп {swap_mb():.0f} МБ, "
          f"свободно {free_pct()}%\n")
    print(f"{'конфигурация':26s} {'ppl':>9s} {'сек':>6s} {'пик свопа':>10s} "
          f"{'мин своб':>9s} {'кеш МБ':>8s}")

    cases = [("bf16 (пол)", QuantConfig(), None),
             ("композит, кеш ВЫКЛ", comp, 0),
             ("композит, кеш 1536 МБ", comp, 1536 << 20)]
    prev = None
    for tag, cfg, budget in cases:
        if budget is not None:
            fake_quant._CACHE_BUDGET = budget
        torch.mps.empty_cache()
        time.sleep(2)
        with Watch() as w:
            t0 = time.time()
            p = perplexity(model, data, cfg)
            dt = time.time() - t0
        st = fake_quant.cache_stats()
        print(f"{tag:26s} {p:9.4f} {dt:6.0f} {w.peak_swap:10.0f} "
              f"{w.min_free:8d}% {st['bytes']/1e6:8.0f}", flush=True)
        if prev is not None:
            print(f"{'':26s} {'':9s} {'':6s} "
                  f"{w.peak_swap - prev:+10.0f}  <- прирост к предыдущей")
        prev = w.peak_swap

    print("\nЧитать так: прирост строки 2 над строкой 1 -- цена ТРАНЗИЕНТОВ "
          "блочного\nпоиска; прирост строки 3 над строкой 2 -- цена "
          "УДЕРЖАНИЯ квантованных копий.")


if __name__ == "__main__":
    main()
