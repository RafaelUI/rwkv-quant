"""
Пересмотр решения "AW вредит proj на 6 битах" (presets.py, сессия 19.07-5).

Почему пересматриваем: решение принято по замеру на eval_corpus_world[:8] --
1016 предсказаний короткого английского. Про этот срез теперь известно, что
он аномально лёгкий (bf16 ppl 11.43 против 16.52 на полном том же корпусе) и
занижает деградацию примерно в шесть раз (tests/eval_corpora_compare.py).
Отдельно: llama.cpp на этой же модели получает от imatrix двукратное
уменьшение потерь (Q4_K_M 7.65% -> 3.44%), то есть для RWKV-7
activation-взвешивание -- главный рычаг, а не второстепенный.

Три режима различаются ДВУМЯ независимыми вещами, и нынешний proj не имеет
ни одной из них -- поэтому просто "включить AW" неинформативно:

  asym_sb6         search=False, ex2=None   <- как сейчас в REDUCTION
  asym_sb6_search  search=True,  ex2=None   <- только грид-поиск scale/min
  asym_sb6_aw      search=True,  ex2=stats  <- поиск + AW-взвешивание

(см. writer.quantize_tensor -> _make_qt_gw_sb6(search=...), ex2 подаётся
только для _aw). Разница между 2-м и 3-м -- чистый вклад AW.

act_stats берутся МУЛЬТИЯЗЫЧНЫЕ и сняты на отложенных чанках -- те же
данные, на которых строился imatrix для llama.cpp, чтобы сравнение
оставалось честным.

Запуск: python tests/ablate_aw_proj.py > /tmp/aw.log 2>&1 &
"""
import copy
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mlx.core as mx  # noqa: E402

from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.presets import REDUCTION  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
OUT_JSON = os.path.expanduser("~/Develop/WKV-kvant/ablate_aw_proj.json")
ACT_STATS = "/tmp/act_stats_1p5b_multiling.pt"
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536


def variant(**mode_overrides):
    cfg = copy.deepcopy(REDUCTION)        # QuantConfig -- не dataclass
    cfg.act_stats_path = ACT_STATS
    cfg.group_scale_mode = dict(cfg.group_scale_mode, **mode_overrides)
    return cfg


CONFIGS = [
    ("bf16", QuantConfig()),
    ("REDUCTION (proj=asym_sb6)", variant()),
    ("proj=asym_sb6_search", variant(proj="asym_sb6_search")),
    ("proj=asym_sb6_aw", variant(proj="asym_sb6_aw")),
]


def tensor_bytes(qt):
    n = 0
    for f in ("dense", "codes", "codes_packed", "scale", "gw_qsqm", "gw_d",
              "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
              "outlier_indices", "outlier_values"):
        t = getattr(qt, f, None)
        if t is not None:
            n += t.numel() * t.element_size()
    return n


def nll_per_seq(model, data, batch_size=1):
    out = []
    for i in range(0, data.shape[0], batch_size):
        b = data[i:i + batch_size]
        logits = model(mx.array(b[:, :-1])).astype(mx.float32)
        tgt = mx.array(b[:, 1:])[..., None]
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, tgt, axis=-1).squeeze(-1)
        mx.eval(nll)
        a = np.array(nll)
        out.extend((float(a[j].sum()), int(a[j].size)) for j in range(a.shape[0]))
        del logits, nll
    return out


def agg(per_seq, langs):
    res = {}
    for k in sorted(set(langs)) + ["ALL"]:
        sel = [x for x, l in zip(per_seq, langs) if k == "ALL" or l == k]
        res[k] = float(np.exp(sum(s for s, _ in sel) / sum(n for _, n in sel)))
    return res


def main():
    blob = torch.load(CORPUS)
    data, langs = blob["tokens"].numpy(), blob["lang"]
    print(f"корпус {data.shape}, act_stats={ACT_STATS} "
          f"exists={os.path.exists(ACT_STATS)}\n", flush=True)

    sd = torch.load(CKPT_PTH, map_location="cpu")
    results = {}
    for name, cfg in CONFIGS:
        t0 = time.time()
        ckpt = QuantizedCheckpoint(
            naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD, head_size=HEAD_SIZE,
            vocab_size=VOCAB, config_repr=repr(cfg),
            tensors={k: quantize_tensor(k, w, cfg, real_gw=True)
                     for k, w in sd.items()})
        mb = sum(tensor_bytes(q) for q in ckpt.tensors.values()) / 1e6
        model = QuantRWKV7(ckpt)
        t1 = time.time()
        res = agg(nll_per_seq(model, data), langs)
        results[name] = {"size_mb": mb, "ppl": res}
        print(f"{name:<28} {mb:7.1f} MB  "
              + "  ".join(f"{k}={v:.4f}" for k, v in res.items())
              + f"   [кв {t1-t0:.0f}s, ppl {time.time()-t1:.0f}s]", flush=True)
        del model, ckpt
        gc.collect()
        mx.clear_cache()

    base = results["bf16"]["ppl"]
    print("\n" + "=" * 76)
    print(f"{'вариант':<28}{'MB':>8}" + "".join(f"{l:>16}" for l in base))
    for name, r in results.items():
        print(f"{name:<28}{r['size_mb']:>8.0f}" + "".join(
            f"{100*(r['ppl'][l]-base[l])/base[l]:>+15.2f}%" for l in base))
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
