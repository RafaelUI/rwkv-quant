"""
Подтверждение находки: группа `small` (k_k, k_a, r_k -- 147K параметров,
0.01% модели) уносит 1.59 из 2.36 п.п. деградации REDUCTION, а по
сербскому 5.52% -> 1.00%. См. tests/ablate_group_contrib.py (STAGE=2).

Почему это правдоподобно: k_k/k_a/r_k -- поканальные модуляторы ВНУТРИ
рекуррентности WKV-7 (нормировка ключа, темп in-context обучения,
receptance-бонус). Их ошибка не портит одно предсказание, а искажает
обновление состояния и копится по длине. Это согласуется с замером
eval_corpora_compare.py: деградация REDUCTION растёт с контекстом
(+1.52% на 128 токенах против +2.57% на 512).

Хранение их в bf16 стоит ~0.15 МБ на 1.5B -- то есть бесплатно.

Проверяются оба пресета плюс объединение с proj=asym_sb6_search
(tests/ablate_aw_proj.py: -0.14 п.п. даром).

Запуск: python tests/confirm_small_fix.py > /tmp/confirm.log 2>&1 &
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
from rwkv_quant.presets import REDUCTION, COMPRESSION  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
OUT_JSON = os.path.expanduser("~/Develop/WKV-kvant/confirm_small_fix.json")
ACT_STATS = "/tmp/act_stats_1p5b_multiling.pt"
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536


def tweak(preset, small=None, proj_mode=None):
    cfg = copy.deepcopy(preset)
    cfg.act_stats_path = ACT_STATS
    if small is not None:
        cfg.bits["small"] = small
    if proj_mode is not None:
        cfg.group_scale_mode = dict(cfg.group_scale_mode, proj=proj_mode)
    return cfg


CONFIGS = [
    ("bf16", QuantConfig()),
    ("REDUCTION", tweak(REDUCTION)),
    ("REDUCTION small=bf16", tweak(REDUCTION, small=16)),
    ("REDUCTION small=bf16 +search", tweak(REDUCTION, small=16,
                                           proj_mode="asym_sb6_search")),
    ("COMPRESSION", tweak(COMPRESSION)),
    ("COMPRESSION small=bf16", tweak(COMPRESSION, small=16)),
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


def nll_per_seq(model, data):
    out = []
    for i in range(data.shape[0]):
        b = data[i:i + 1]
        logits = model(mx.array(b[:, :-1])).astype(mx.float32)
        tgt = mx.array(b[:, 1:])[..., None]
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, tgt, axis=-1).squeeze(-1)
        mx.eval(nll)
        a = np.array(nll)
        out.append((float(a.sum()), int(a.size)))
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
        res = agg(nll_per_seq(model, data), langs)
        results[name] = {"size_mb": mb, "ppl": res}
        print(f"{name:<32}{mb:8.2f} MB  "
              + "  ".join(f"{k}={v:.4f}" for k, v in res.items())
              + f"   [{time.time()-t0:.0f}s]", flush=True)
        del model, ckpt
        gc.collect()
        mx.clear_cache()

    base = results["bf16"]["ppl"]
    print("\n" + "=" * 84)
    print(f"{'вариант':<32}{'MB':>9}" + "".join(f"{l:>11}" for l in base))
    for name, r in results.items():
        print(f"{name:<32}{r['size_mb']:>9.1f}" + "".join(
            f"{100*(r['ppl'][l]-base[l])/base[l]:>+10.2f}%" for l in base))
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
