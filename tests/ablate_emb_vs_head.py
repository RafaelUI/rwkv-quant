"""emb против head: стоит ли им быть одной группой.

Мотивация. Пока emb и head сидели в группе `emb_head`, битность
выбиралась по худшему из двух, а вместе это 29% параметров модели
(65536 x D на каждый). Побочный результат замера развилки QLoRA-базы
(NEXT_SESSION) утверждает, что КВАНТОВАНИЕ emb не стоит ничего по
качеству на обоих масштабах; про head такого замера нет.

Здесь -- прямая изолированная проверка: квантуется РОВНО одна из двух
таблиц, всё остальное bf16. Если чувствительность различается, общая
группа обходится в биты на ровном месте.

Считается и цена: sb6 стоит bits+0.5 бит/вес (измерено,
tests/probe_schema_cost.py), поэтому экономия от снижения битности emb
на два бита -- это 2 x n_emb / 8 байт, и её видно сразу в МБ.

  RWKVQ_CKPT    чекпоинт      (умолчание -- 0.1B)
  RWKVQ_NSEQ    послед.       (8)
  RWKVQ_SEQLEN  длина         (256)
  RWKVQ_OUT     json          (/tmp/emb_vs_head.json)

Замечание о масштабе (закон 10): вывод с одного чекпоинта НЕ переносится
на другой. Гонять на 0.1B ради механики, решение принимать по 1.5B и 2.9B.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
ACT = os.environ.get("RWKVQ_ACT_STATS") or None
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 8))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 256))
OUT = os.environ.get("RWKVQ_OUT", "/tmp/emb_vs_head.json")

MODE = os.environ.get("RWKVQ_MODE", "asym_sb6_aw" if ACT else "asym_sb6")


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"],
                          capture_output=True, text=True).stdout.strip()


def one(group, bits):
    return QuantConfig(**{group: bits}, group_scale={group: 32},
                       group_scale_mode={group: MODE}, act_stats_path=ACT)


def main():
    print(f"своп до: {swap()}", flush=True)
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")
    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    n_emb = sd["emb.weight"].numel()
    n_head = sd["head.weight"].numel()
    n_tot = sum(t.numel() for t in sd.values())
    print(f"emb {tuple(sd['emb.weight'].shape)} = {n_emb/1e6:.1f}M, "
          f"head {tuple(sd['head.weight'].shape)} = {n_head/1e6:.1f}M, "
          f"вместе {(n_emb+n_head)/n_tot*100:.1f}% модели")
    print(f"режим блочной схемы: {MODE}, act_stats={'есть' if ACT else 'нет'}\n")

    base = perplexity(model, data, QuantConfig())
    print(f"BASELINE bf16 ppl={base:.4f}\n")
    print(f"{'что квантуем':16s} {'бит':>4s} {'ppl':>11s} {'Δ%':>9s} "
          f"{'МБ группы':>10s}")
    rows = []
    for group, n in (("emb", n_emb), ("head", n_head)):
        for bits in (6, 5, 4):
            t0 = time.time()
            p = perplexity(model, data, one(group, bits))
            delta = 100 * (p - base) / base
            mb = n * (bits + 0.5) / 8 / 1e6
            rows.append({"group": group, "bits": bits, "ppl": p,
                         "delta_pct": delta, "mb": mb})
            print(f"{group:16s} {bits:4d} {p:11.4f} {delta:+9.2f}% "
                  f"{mb:10.1f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\nразница чувствительности при равной битности:")
    for bits in (6, 5, 4):
        e = next(r["delta_pct"] for r in rows if r["group"] == "emb" and r["bits"] == bits)
        h = next(r["delta_pct"] for r in rows if r["group"] == "head" and r["bits"] == bits)
        print(f"  {bits} бит: emb {e:+.2f}%  head {h:+.2f}%  "
              f"head дороже в {h/e if e else float('nan'):.1f}x")

    e6 = next(r["mb"] for r in rows if r["group"] == "emb" and r["bits"] == 6)
    e4 = next(r["mb"] for r in rows if r["group"] == "emb" and r["bits"] == 4)
    print(f"\nцена вопроса: emb 6->4 бит освобождает {e6-e4:.1f} МБ "
          f"на этом чекпоинте")

    with open(OUT, "w") as f:
        json.dump({"ckpt": CKPT, "baseline": base, "mode": MODE,
                   "n_emb": n_emb, "n_head": n_head, "n_total": n_tot,
                   "n_pred": int(data.shape[0] * (data.shape[1] - 1)),
                   "rows": rows}, f, indent=1)
    print(f"-> {OUT}")
    print(f"своп после: {swap()}")


if __name__ == "__main__":
    main()
