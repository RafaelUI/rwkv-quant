"""Композитная проверка: даёт ли разделение emb/head что-то на РЕАЛЬНОМ
пресете, а не в изоляции.

tests/ablate_emb_vs_head.py показал изолированную чувствительность
(1.5B, 19456 предсказаний: emb@5 -0.02%, head@5 +0.67%). Но
изолированный результат НЕ переносится в композит (закон 5), поэтому
здесь меряется пресет целиком с разной битностью emb при неизменном
head, и рядом считается реальный размер файла.

act_stats_path принудительно None, а не путь из пресета. Причина: файл
в /tmp не переживает перезагрузку, и при его отсутствии asym_sb6_aw
молча вырождается в asym_sb6_search -- то есть один и тот же вызов даёт
разный артефакт в зависимости от содержимого /tmp. Для сравнения
вариантов между собой это недопустимо; фиксируем режим явно.

  RWKVQ_CKPT   чекпоинт (умолчание -- 1.5B)
  RWKVQ_PRESET reduction | compression
"""
import copy
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
from rwkv_quant.presets import PRESETS

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
PRESET = os.environ.get("RWKVQ_PRESET", "reduction")
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 38))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))
OUT = os.environ.get("RWKVQ_OUT", f"/tmp/confirm_emb_split_{PRESET}.json")
EMB_BITS = [int(x) for x in os.environ.get("RWKVQ_EMB_BITS", "6,5,4").split(",")]


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"],
                          capture_output=True, text=True).stdout.strip()


def variant(base, emb_bits):
    c = QuantConfig(group_scale=dict(base.group_scale),
                    group_scale_mode=dict(base.group_scale_mode),
                    clip_percentiles=dict(base.clip_percentiles),
                    outlier_fracs=dict(base.outlier_fracs),
                    bits_overrides=dict(base.bits_overrides),
                    act_stats_path=None,          # см. докстринг
                    **dict(base.bits))
    c.bits["emb"] = emb_bits
    return c


def est_mb(sd, cfg):
    from rwkv_quant.calibration import schema_space as _ss
    from rwkv_quant.calibration.outlier_scan import GROUP_KEY_PATTERNS
    tot = 0
    for key, t in sd.items():
        n, grp = t.numel(), None
        for g, pats in GROUP_KEY_PATTERNS.items():
            if any(key.endswith(p) or p in key for p in pats):
                grp = g
                break
        bits = cfg.bits.get(grp, 16) if grp else 16
        if grp is None or t.dim() < 2 or bits >= 16:
            tot += n * 16
            continue
        gs = cfg.group_scale.get(grp)
        mode = cfg.group_scale_mode.get(grp, "")
        eff = (_ss.SB6_COST[bits] if gs and mode.startswith("asym_sb6")
               else _ss.ASYM_COST if gs else _ss._rtn_cost(bits, int(t.shape[-1])))
        tot += n * eff
    return tot / 8 / 1e6


def main():
    print(f"своп до: {swap()}", flush=True)
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")
    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    base = PRESETS[PRESET]
    print(f"пресет {PRESET}: {base}")
    print(f"корпус {tuple(data.shape)} = {data.shape[0]*(data.shape[1]-1)} предсказаний\n")

    bf16 = perplexity(model, data, QuantConfig())
    print(f"BASELINE bf16 ppl={bf16:.4f}\n")
    print(f"{'emb бит':>8s} {'head бит':>9s} {'ppl':>10s} {'Δ%':>8s} "
          f"{'МБ':>8s} {'ΔМБ':>7s}")
    rows, ref_mb = [], None
    for eb in EMB_BITS:
        cfg = variant(base, eb)
        t0 = time.time()
        p = perplexity(model, data, cfg)
        delta = 100 * (p - bf16) / bf16
        mb = est_mb(sd, cfg)
        if ref_mb is None:
            ref_mb = mb
        rows.append({"emb_bits": eb, "head_bits": cfg.bits["head"],
                     "ppl": p, "delta_pct": delta, "mb": mb})
        print(f"{eb:8d} {cfg.bits['head']:9d} {p:10.4f} {delta:+8.2f}% "
              f"{mb:8.1f} {mb-ref_mb:+7.1f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\nвывод по этому чекпоинту:")
    r0 = rows[0]
    for r in rows[1:]:
        dd = r["delta_pct"] - r0["delta_pct"]
        dm = r0["mb"] - r["mb"]
        print(f"  emb {r0['emb_bits']}->{r['emb_bits']}: "
              f"качество {dd:+.2f} п.п., размер -{dm:.1f} МБ"
              + ("  (даром)" if dd <= 0.05 else ""))

    with open(OUT, "w") as f:
        json.dump({"ckpt": CKPT, "preset": PRESET, "bf16_ppl": bf16,
                   "n_pred": int(data.shape[0] * (data.shape[1] - 1)),
                   "rows": rows}, f, indent=1)
    print(f"-> {OUT}")
    print(f"своп после: {swap()}")


if __name__ == "__main__":
    main()
