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

ОДИН КОНФИГ НА ПРОЦЕСС (закон 13). Без аргументов скрипт гоняет всю
таблицу в одном процессе -- это допустимо на 1.5B и НЕ допустимо на
2.9B, где модель занимает 5.9 ГБ и остаточная память от предыдущего
конфига решает исход. С аргументом гоняется ровно один вариант, а
результаты накапливаются в JSON:

    for v in bf16 6 5; do python tests/confirm_emb_split.py $v; done

  RWKVQ_CKPT   чекпоинт (умолчание -- 1.5B)
  RWKVQ_PRESET reduction | compression
  RWKVQ_OUT    json, дописывается между процессами
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
# batch_size=1 обязателен на 2.9B: модель занимает 5.9 ГБ на устройстве,
# и логиты [2, 511, 65536] уже не влезают -- MPS роняет command buffer и
# возвращает НУЛИ, а не ошибку (защита -- в ablation.perplexity).
BATCH = int(os.environ.get("RWKVQ_BATCH", 2))
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


def load_json():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"ckpt": CKPT, "preset": PRESET, "rows": []}


def save_json(doc):
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)


def report(doc):
    rows = sorted(doc["rows"], key=lambda r: -r["emb_bits"])
    bf16 = doc.get("bf16_ppl")
    print(f"\n{'emb бит':>8s} {'head бит':>9s} {'ppl':>10s} {'Δ%':>8s} "
          f"{'МБ':>8s} {'ΔМБ':>7s}")
    ref_mb = rows[0]["mb"] if rows else 0
    for r in rows:
        d = (100 * (r["ppl"] - bf16) / bf16) if bf16 else float("nan")
        print(f"{r['emb_bits']:8d} {r['head_bits']:9d} {r['ppl']:10.4f} "
              f"{d:+8.2f}% {r['mb']:8.1f} {r['mb']-ref_mb:+7.1f}")
    if bf16 and len(rows) > 1:
        print("\nвывод по этому чекпоинту:")
        r0 = rows[0]
        d0 = 100 * (r0["ppl"] - bf16) / bf16
        for r in rows[1:]:
            dd = 100 * (r["ppl"] - bf16) / bf16 - d0
            print(f"  emb {r0['emb_bits']}->{r['emb_bits']}: "
                  f"качество {dd:+.2f} п.п., размер -{r0['mb']-r['mb']:.1f} МБ"
                  + ("  (даром)" if dd <= 0.05 else ""))


def run_one(which):
    print(f"своп до: {swap()}", flush=True)
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")
    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    print(f"своп после загрузки модели: {swap()}", flush=True)
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    base = PRESETS[PRESET]
    n_pred = int(data.shape[0] * (data.shape[1] - 1))
    print(f"пресет {PRESET}, корпус {tuple(data.shape)} = {n_pred} предсказаний")

    doc = load_json()
    doc["n_pred"] = n_pred
    t0 = time.time()
    if which == "bf16":
        p = perplexity(model, data, QuantConfig(), batch_size=BATCH)
        doc["bf16_ppl"] = p
        print(f"bf16 ppl={p:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    else:
        eb = int(which)
        cfg = variant(base, eb)
        p = perplexity(model, data, cfg, batch_size=BATCH)
        mb = est_mb(sd, cfg)
        doc["rows"] = [r for r in doc["rows"] if r["emb_bits"] != eb]
        doc["rows"].append({"emb_bits": eb, "head_bits": cfg.bits["head"],
                            "ppl": p, "mb": mb})
        bf = doc.get("bf16_ppl")
        dd = f"{100*(p-bf)/bf:+.2f}%" if bf else "(bf16 ещё не мерен)"
        print(f"emb={eb} head={cfg.bits['head']} ppl={p:.4f} Δ={dd} "
              f"{mb:.1f} МБ  [{time.time()-t0:.0f}s]", flush=True)
    save_json(doc)
    report(doc)
    print(f"-> {OUT}\nсвоп после: {swap()}")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--report":
            report(load_json())
            return
        run_one(sys.argv[1])
        return
    # Намеренно НЕ гоняем всё в одном процессе. Модель загружается в
    # run_one, и повторная загрузка в том же процессе оставляет прежнюю
    # в памяти устройства -- на 2.9B это +5.9 ГБ за конфиг. Драйвер
    # обязан быть снаружи.
    print(__doc__)
    print("Использование: confirm_emb_split.py <bf16|6|5|4>  |  --report\n"
          "Пример:  for v in bf16 6 5; do "
          "python tests/confirm_emb_split.py $v; done")


if __name__ == "__main__":
    main()
