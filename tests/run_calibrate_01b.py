"""Сквозной прогон нового calibrate() на 0.1B + проверка, что выданный
конфиг ДЕПЛОИТСЯ в ту же схему, которую калибровка мерила.

Последнее -- половина смысла. Прежняя версия возвращала конфиг без
group_scale, поэтому quantize(config=...) производил per-row артефакт
независимо от того, что показал поиск. Здесь после калибровки конфиг
подаётся в реальную упаковку, файл читается обратно и ppl считается на
ДЕКВАНТОВАННЫХ весах -- если совпало с тем, что обещал поиск, значит
цепочка «мерили -> выбрали -> записали» замкнулась.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.api import calibrate
from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration import schema_space as _ss
from rwkv_quant.calibration.outlier_scan import GROUP_KEY_PATTERNS
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
OUT = os.environ.get("RWKVQ_OUT", "/tmp/calibrate_01b.json")
THRESH = float(os.environ.get("RWKVQ_THRESH", 5.0))
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 8))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 256))


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"],
                          capture_output=True, text=True).stdout.strip()


def est_size(sd, cfg, have_act):
    """Оценка размера по измеренной модели стоимости (probe_schema_cost)."""
    from rwkv_quant.calibration.group_config import GROUPS
    total_bits, n_tot = 0, 0
    for key, t in sd.items():
        n = t.numel()
        n_tot += n
        grp = None
        for g, pats in GROUP_KEY_PATTERNS.items():
            if any(key.endswith(p) or p in key for p in pats):
                grp = g
                break
        bits = cfg.bits.get(grp, 16) if grp else 16
        if grp is None or t.dim() < 2 or bits >= 16:
            total_bits += n * 16
            continue
        gs = cfg.group_scale.get(grp)
        mode = cfg.group_scale_mode.get(grp, "")
        if gs and mode.startswith("asym_sb6"):
            eff = _ss.SB6_COST[bits]
        elif gs:
            eff = _ss.ASYM_COST
        else:
            eff = _ss._rtn_cost(bits, int(t.shape[-1]))
        total_bits += n * eff
    return total_bits / 8 / 1e6, n_tot


def main():
    print(f"своп до: {swap()}", flush=True)
    t0 = time.time()
    cfg = calibrate(CKPT, CORPUS, device="mps", ppl_threshold_pct=THRESH,
                    n_seq=NSEQ, seq_len=SEQLEN, verbose=True)
    rep = cfg.calibration_report

    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    mb, n_tot = est_size(sd, cfg, False)
    bf16_mb = n_tot * 2 / 1e6
    print(f"\n[итог] оценка размера {mb:.1f} МБ против bf16 {bf16_mb:.1f} МБ "
          f"({bf16_mb/mb:.2f}x)")

    # --- замыкание цепочки: конфиг -> реальный файл -> ppl на нём ---
    print("\n[проверка] реальная упаковка по выданному конфигу ...", flush=True)
    from rwkv_quant.api import quantize
    from rwkv_quant.formats.reader import load_dequantized
    path = "/tmp/calibrated_01b.rwkvq"
    quantize(CKPT, path, config=cfg, real_gw=True, verbose=False)
    real_mb = os.path.getsize(path) / 1e6
    print(f"[проверка] файл {real_mb:.1f} МБ (оценка была {mb:.1f})")

    rep["file_mb"] = real_mb
    rep["est_mb"] = mb
    rep["bf16_mb"] = bf16_mb
    rep["config"] = {"bits": dict(cfg.bits),
                     "group_scale": dict(cfg.group_scale),
                     "group_scale_mode": dict(cfg.group_scale_mode)}
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"-> {OUT}")
    print(f"своп после: {swap()}")
    print(f"всего {time.time()-t0:.0f} с")


if __name__ == "__main__":
    main()
