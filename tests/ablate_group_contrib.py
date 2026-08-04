"""
Вклад каждой группы в деградацию REDUCTION (leave-one-out).

Вопрос: +2.36% на мультиязычном корпусе -- чьи это проценты? Каждая
группа по очереди возвращается в bf16, остальные остаются как в пресете.
Насколько упал Δ -- столько группа и уносила. Сумма вкладов не обязана
давать 2.36%: эффекты не аддитивны (см. presets.py -- четыре LoRA-ветки
на INT4 порознь безобидны, вместе дают ~150x взрыв).

Размер печатается рядом, потому что решение всегда компромисс:
группа, которая уносит много процентов и при этом мала, -- очевидный
кандидат на повышение битности; большая и безобидная -- кандидат на
понижение.

Отдельно проверяется гипотеза о структуре (см. GGUF_COMPARE.md): наш
sb6 -- аналог block_q4_K (асимметрия, блок 32, 6-битные scale/min), а
llama.cpp на шести битах использует block_q6_K (СИММЕТРИЧНО, блок 16,
8-битный scale) и получает +0.41% против наших +2.36% при том же
бюджете бит. Здесь мерится только вклад групп; структурные варианты --
следующим шагом, на той группе, которая окажется главной.

Запуск: python tests/ablate_group_contrib.py > /tmp/contrib.log 2>&1 &
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
OUT_JSON = os.path.expanduser("~/Develop/WKV-kvant/ablate_group_contrib.json")
ACT_STATS = "/tmp/act_stats_1p5b_multiling.pt"
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536


def dequant_groups(*groups):
    """REDUCTION, но перечисленные группы -- bf16 (bits=16 отключает и
    gw-ветку: writer идёт в gw только при bits < 16)."""
    cfg = copy.deepcopy(REDUCTION)
    cfg.act_stats_path = ACT_STATS
    for g in groups:
        cfg.bits[g] = 16
    return cfg


LORAS = ("w_lora", "a_lora", "v_lora", "g_lora", "small")

STAGE = os.environ.get("STAGE", "1")

if STAGE == "1":
    CONFIGS = [
        ("bf16", QuantConfig()),
        ("REDUCTION", dequant_groups()),
        ("  проекции proj -> bf16", dequant_groups("proj")),
        ("  cmix -> bf16", dequant_groups("cmix")),
        ("  emb_head -> bf16", dequant_groups("emb_head")),
        ("  все lora+small -> bf16", dequant_groups(*LORAS)),
    ]
else:
    # Этап 2: связка lora+small уносит 1.71 из 2.36 п.п. при том, что
    # весит 47 МБ из 1255. Дробим по одной группе, чтобы найти виновную.
    CONFIGS = [("bf16", QuantConfig()), ("REDUCTION", dequant_groups())] + [
        (f"  {g} -> bf16", dequant_groups(g)) for g in LORAS]


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
        print(f"{name:<26}{mb:8.1f} MB  "
              + "  ".join(f"{k}={v:.4f}" for k, v in res.items())
              + f"   [{time.time()-t0:.0f}s]", flush=True)
        del model, ckpt
        gc.collect()
        mx.clear_cache()

    base = results["bf16"]["ppl"]
    red = results["REDUCTION"]["ppl"]
    print("\n" + "=" * 88)
    print(f"{'вариант':<26}{'MB':>8}" + "".join(f"{l:>11}" for l in base)
          + f"{'вклад группы (ALL)':>22}")
    d_red = 100 * (red["ALL"] - base["ALL"]) / base["ALL"]
    for name, r in results.items():
        d = "".join(f"{100*(r['ppl'][l]-base[l])/base[l]:>+10.2f}%" for l in base)
        # сколько процентных пунктов деградации ушло вместе с группой
        contrib = ""
        if name.startswith("  "):
            d_var = 100 * (r["ppl"]["ALL"] - base["ALL"]) / base["ALL"]
            contrib = f"{d_red - d_var:>+16.2f} п.п."
        print(f"{name:<26}{r['size_mb']:>8.0f}{d}{contrib}")
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
