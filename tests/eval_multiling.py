"""
ppl по языкам (ru/en/sr) на реальном пути QuantRWKV7 (MLX), 1.5B.

Зачем: пресеты калибровались на англоязычном eval_corpus_world, а
приложение мультиязычное. World-vocab кодирует кириллицу вдвое плотнее
латиницы (2.5 против 4.5 символов на токен), то есть на кириллицу
приходится больше редких токенов -- а по редким токенам квантование
emb/head бьёт сильнее всего. Агрегированный ppl это усредняет и прячет.

Меряется ПАРНО: одни и те же последовательности на bf16 и на каждом
пресете, сравнивается Δppl% на язык. Абсолютный ppl на 9 окнах (en/sr)
шумноват, но парная разность на идентичных данных -- нет.

Квантование in-memory через writer.quantize_tensor(real_gw=True), то есть
РЕАЛЬНАЯ упаковка sb6 и реальный кернель, а не fake-dequant: fake считает
ту же арифметику ошибки, но держит веса плотными в bf16 (на 1.5B это
лишние 3 ГБ RAM и другой путь исполнения).

Запуск (долгий, ставить в фон):
    python tests/eval_multiling.py [act_stats_path] > /tmp/eval_multiling.log 2>&1 &
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
OUT_JSON = os.path.expanduser("~/Develop/WKV-kvant/eval_multiling_1p5b.json")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

ACT_STATS = (sys.argv[1] if len(sys.argv) > 1
             else "/tmp/act_stats_1p5b_multiling.pt")


def with_stats(cfg: QuantConfig) -> QuantConfig:
    """Пресеты жёстко указывают act_stats_path=/tmp/act_stats_1p5b.pt,
    который не переживает перезагрузку. Подставляем свой (снятый на
    ОТЛОЖЕННЫХ чанках, см. build_eval_multiling.py), иначе AW-режимы молча
    вырождаются в поиск без взвешивания и цифры будут не про пресет.

    QuantConfig -- обычный класс, не dataclass, поэтому копия глубокая:
    пресеты в presets.py -- модульные синглтоны, мутировать их на месте
    значит менять поведение всего, что импортирует presets."""
    out = copy.deepcopy(cfg)
    out.act_stats_path = ACT_STATS
    return out


CONFIGS = [
    ("bf16 baseline", QuantConfig()),
    ("REDUCTION", with_stats(REDUCTION)),
    ("COMPRESSION", with_stats(COMPRESSION)),
]


def tensor_bytes(qt) -> int:
    """Честный размер одного QuantizedTensor во всех режимах формата.
    (measure_size_mb в quality_speed_curve.py считает только v1-путь
    codes+scale и падает/врёт на sb6 и на упакованных нибблах.)"""
    n = 0
    for field in ("dense", "codes", "codes_packed", "scale", "gw_qsqm",
                  "gw_d", "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
                  "outlier_indices", "outlier_values"):
        t = getattr(qt, field, None)
        if t is not None:
            n += t.numel() * t.element_size()
    return n


def checkpoint_mb(ckpt) -> float:
    return sum(tensor_bytes(qt) for qt in ckpt.tensors.values()) / 1e6


def build_in_memory(sd, cfg):
    tensors = {}
    for key, w in sd.items():
        tensors[key] = quantize_tensor(key, w, cfg, real_gw=True)
    return QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                               head_size=HEAD_SIZE, vocab_size=VOCAB,
                               tensors=tensors, config_repr=repr(cfg))


def nll_per_seq(model, data_np, batch_size=1):
    """Возвращает (sum_nll, n_tok) на каждую последовательность отдельно --
    чтобы потом сгруппировать по языку без повторного прогона модели.

    NLL считается через logsumexp, а НЕ через материализацию log_softmax:
    logits тут [B, 511, 65536], и полный fp32-logp плюс промежуточные копии
    softmax/log -- это ~0.8 ГБ транзиента на батч из 2. На 16 ГБ вместе с
    bf16 baseline (sd 3 ГБ + ckpt 3 ГБ + модель в MLX 3 ГБ) этого хватает,
    чтобы уйти в своп. lse - picked даёт то же число без лишних копий."""
    out = []
    for i in range(0, data_np.shape[0], batch_size):
        batch = data_np[i:i + batch_size]
        logits = model(mx.array(batch[:, :-1])).astype(mx.float32)
        tgt = mx.array(batch[:, 1:].astype(np.int32))[..., None]
        picked = mx.take_along_axis(logits, tgt, axis=-1).squeeze(-1)
        nll = mx.logsumexp(logits, axis=-1) - picked          # [B, T]
        mx.eval(nll)
        nll_np = np.array(nll)
        for b in range(nll_np.shape[0]):
            out.append((float(nll_np[b].sum()), int(nll_np[b].size)))
        del logits, picked, nll
        print(f"    ppl {min(i+batch_size, data_np.shape[0])}/{data_np.shape[0]}",
              flush=True)
    return out


def ppl_grouped(per_seq, keys):
    """ppl по произвольной группировке (язык или id чанка) + ALL."""
    res = {}
    for k in sorted(set(keys), key=str) + ["ALL"]:
        sel = [(s, n) for (s, n), kk in zip(per_seq, keys)
               if k == "ALL" or kk == k]
        res[str(k)] = float(np.exp(sum(s for s, _ in sel) / sum(n for _, n in sel)))
    return res


def main():
    if not os.path.exists(ACT_STATS):
        print(f"!! нет {ACT_STATS} -- AW-режимы выродятся, цифры будут "
              f"не про пресеты. Сначала collect_act_stats.py", flush=True)
    blob = torch.load(CORPUS)
    data = blob["tokens"].numpy()
    langs = blob["lang"]
    chunk_ids = blob["chunk"]
    preview = blob.get("preview", {})
    print(f"корпус: {data.shape}, языки: "
          f"{ {l: langs.count(l) for l in sorted(set(langs))} }", flush=True)

    print("загрузка 1.5B ...", flush=True)
    t0 = time.time()
    sd = torch.load(CKPT_PTH, map_location="cpu")
    print(f"  {len(sd)} тензоров за {time.time()-t0:.0f}s", flush=True)

    results = {}
    for name, cfg in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        ckpt = build_in_memory(sd, cfg)
        mb = checkpoint_mb(ckpt)
        print(f"  квантование {time.time()-t0:.0f}s, размер {mb:.1f} MB", flush=True)

        model = QuantRWKV7(ckpt)
        t0 = time.time()
        per_seq = nll_per_seq(model, data)
        res = ppl_grouped(per_seq, langs)
        print(f"  ppl за {time.time()-t0:.0f}s: "
              + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
        results[name] = {"size_mb": mb, "ppl": res,
                         "ppl_chunk": ppl_grouped(per_seq, chunk_ids),
                         "nll": per_seq}

        del model, ckpt
        gc.collect()
        mx.clear_cache()

    base = results["bf16 baseline"]["ppl"]
    print("\n" + "=" * 64)
    print(f"{'конфиг':<16}{'MB':>8}   " + "".join(f"{l:>14}" for l in base))
    for name, r in results.items():
        row = "".join(
            f"{r['ppl'][l]:>8.3f}({100*(r['ppl'][l]-base[l])/base[l]:+.2f}%)"
            for l in base)
        print(f"{name:<16}{r['size_mb']:>8.0f}   {row}")
    # Разбивка по документам. Агрегат по языку скрывает, идёт ли деградация
    # от языка или от одного нетипичного текста: у sr в измерительной части
    # всего два чанка, и без этой таблицы их не различить.
    n_win = {c: chunk_ids.count(c) for c in set(chunk_ids)}
    print("\nпо чанкам:")
    hdr = "".join(f"{n:>22}" for n in results if n != "bf16 baseline")
    print(f"{'чанк':<6}{'lang':<5}{'окон':>5}{'bf16':>9}{hdr}")
    for c in sorted(set(chunk_ids)):
        lang = langs[chunk_ids.index(c)]
        b = results["bf16 baseline"]["ppl_chunk"][str(c)]
        row = "".join(
            f"{r['ppl_chunk'][str(c)]:>13.3f}({100*(r['ppl_chunk'][str(c)]-b)/b:+6.2f}%)"
            for n, r in results.items() if n != "bf16 baseline")
        print(f"{c:<6}{lang:<5}{n_win[c]:>5}{b:>9.3f}{row}   {preview.get(c,'')[:44]}")

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
