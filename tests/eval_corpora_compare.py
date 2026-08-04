"""
Откуда взялось расхождение: README даёт REDUCTION +0.12%, а
tests/eval_multiling.py на мультиязычном корпусе -- +2.36%.

Кандидатов три, и они разделяются одним прогоном, потому что модель
строится ОДИН раз на конфиг, а корпусов прогоняется четыре:

  A world[:8]   -- ровно то, на чём считались числа README (8 x 128,
                   англ.). Если A воспроизводит 11.430 / 11.4438 / 11.710,
                   пайплайн эквивалентен опубликованному и дальше можно
                   верить разностям.
  B world[:24]  -- тот же текст и та же длина, но выборка втрое больше:
                   A vs B = вклад РАЗМЕРА ВЫБОРКИ (1016 предсказаний --
                   это ~2 предложения, ppl на них шумный).
  C ml@128      -- мультиязычный корпус, обрезанный до 128 токенов:
                   B vs C = вклад ЯЗЫКА при равной длине контекста.
  D ml@512      -- он же целиком: C vs D = вклад ДЛИНЫ КОНТЕКСТА
                   (на 512 токенах рекуррентное состояние успевает
                   накопить ошибку квантования, на 128 -- нет).

act_stats берутся английские (/tmp/act_stats_1p5b.pt, рецепт
collect_act_stats.py по умолчанию: world[8:16]) -- ровно как при
калибровке пресетов, иначе сравнение с README нечестное.

Запуск: python tests/eval_corpora_compare.py > /tmp/cmp.log 2>&1 &
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
WORLD = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
ML = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
OUT_JSON = os.path.expanduser("~/Develop/WKV-kvant/eval_corpora_compare.json")
ACT_STATS = "/tmp/act_stats_1p5b.pt"
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536


def with_stats(cfg):
    out = copy.deepcopy(cfg)          # QuantConfig -- не dataclass
    out.act_stats_path = ACT_STATS
    return out


CONFIGS = [("bf16", QuantConfig()),
           ("REDUCTION", with_stats(REDUCTION)),
           ("COMPRESSION", with_stats(COMPRESSION))]


def load_corpora():
    w = torch.load(WORLD).numpy().astype(np.int32)
    ml = torch.load(ML)
    mlt = ml["tokens"].numpy().astype(np.int32)
    langs = ml["lang"]
    return [
        ("A world[:8]@128", w[:8], None),
        ("B world[:24]@128", w, None),
        ("C ml@128", mlt[:, :128], langs),
        ("D ml@512", mlt, langs),
    ]


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


def agg(per_seq, keys=None):
    res = {"ALL": float(np.exp(sum(s for s, _ in per_seq)
                               / sum(n for _, n in per_seq)))}
    for k in sorted(set(keys or [])):
        sel = [x for x, kk in zip(per_seq, keys) if kk == k]
        res[k] = float(np.exp(sum(s for s, _ in sel) / sum(n for _, n in sel)))
    return res


def main():
    corpora = load_corpora()
    for n, d, _ in corpora:
        print(f"{n}: {d.shape} -> {d.shape[0]*(d.shape[1]-1)} предсказаний")
    print(f"act_stats: {ACT_STATS} exists={os.path.exists(ACT_STATS)}\n", flush=True)

    sd = torch.load(CKPT_PTH, map_location="cpu")
    results = {}
    for name, cfg in CONFIGS:
        t0 = time.time()
        ckpt = QuantizedCheckpoint(
            naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD, head_size=HEAD_SIZE,
            vocab_size=VOCAB, config_repr=repr(cfg),
            tensors={k: quantize_tensor(k, w, cfg, real_gw=True)
                     for k, w in sd.items()})
        model = QuantRWKV7(ckpt)
        print(f"=== {name} (квантование {time.time()-t0:.0f}s) ===", flush=True)
        results[name] = {}
        for cname, data, keys in corpora:
            t0 = time.time()
            results[name][cname] = agg(nll_per_seq(model, data), keys)
            print(f"  {cname:<18} " + "  ".join(
                f"{k}={v:.4f}" for k, v in results[name][cname].items())
                + f"   [{time.time()-t0:.0f}s]", flush=True)
        del model, ckpt
        gc.collect()
        mx.clear_cache()

    print("\n" + "=" * 78)
    print(f"{'корпус':<18}{'предск.':>9}{'bf16':>10}"
          f"{'REDUCTION':>22}{'COMPRESSION':>22}")
    for cname, data, _ in corpora:
        b = results["bf16"][cname]["ALL"]
        row = ""
        for cfg_name in ("REDUCTION", "COMPRESSION"):
            v = results[cfg_name][cname]["ALL"]
            row += f"{v:>13.4f}({100*(v-b)/b:+6.2f}%)"
        print(f"{cname:<18}{data.shape[0]*(data.shape[1]-1):>9}{b:>10.4f}{row}")
    print("\nREADME для сверки с A: bf16 11.430  REDUCTION 11.4438 (+0.12%)  "
          "COMPRESSION 11.710 (+2.4%)")
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
