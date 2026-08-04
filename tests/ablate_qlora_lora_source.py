"""
Насколько сдвинется QLoRA-база, если LoRA-ветки брать из .rwkvq, а не
из .pth.

Развилка. QLoRA в rwkv-metal квантует ТОЛЬКО proj / cmix / head, а
w/a/v/g_lora, emb, small и нормы берёт из исходного bf16-чекпоинта.
После шага 4 .rwkvq самодостаточен, и .pth из тренировочного пути можно
убрать -- но тогда LoRA-ветки приедут ДЕКВАНТОВАННЫМИ (в REDUCTION они
asym-gw64 @6, g_lora -- per-row @8). База сдвинется. Вопрос: насколько.

Почему нельзя взять готовое число. `ablate_group_contrib.py` мерил
leave-one-out на ПОЛНОМ REDUCTION и дал по веткам +0.02...+0.09 п.п.
каждая. Это не ответ: во-первых, там квантовано всё остальное, включая
emb, а QLoRA-база другая; во-вторых, закон 5 -- сумма изолированных
вкладов не переносится в композит. Нужен прямой замер именно того
состава, который получится.

Составы (различаются РОВНО LoRA-ветками, всё прочее идентично):

  qlora_pth     как сейчас: proj/cmix/head @6 sb6, LoRA/emb/small -- bf16
  qlora_wav     + w/a/v_lora из .rwkvq (asym-gw64 @6)
  qlora_all     + g_lora из .rwkvq (per-row @8)  <- то, что получится
  REDUCTION     весь пресет целиком, включая emb -- якорь: цифра должна
                сойтись с таблицей в NEXT_SESSION (+0.77% ALL на 1.5B)

bf16 -- общий ноль. Корпус мультиязычный, 512 токенов (закон 9).

    python tests/ablate_qlora_lora_source.py [конфиг ...] > /tmp/qlora_src.log 2>&1 &

Без аргументов гоняет всё. На 2.9B запускать ПО ОДНОМУ конфигу на
процесс (законы 2, 11, 13); чекпоинт берётся из RWKVQ_CKPT.
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
from rwkv_quant.formats.reader import _dequantize_one  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from eval_multiling import nll_per_seq, ppl_grouped, checkpoint_mb  # noqa: E402

CKPT_PTH = os.environ.get(
    "RWKVQ_CKPT",
    os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
ACT_STATS = os.environ.get("RWKVQ_ACT_STATS",
                           "/tmp/act_stats_1p5b_multiling.pt")
OUT_JSON = os.environ.get(
    "RWKVQ_OUT",
    os.path.expanduser("~/Develop/WKV-kvant/qlora_lora_source.json"))

LORA_GROUPS = ("w_lora", "a_lora", "v_lora", "g_lora")


def _qlora(**bits) -> QuantConfig:
    """База QLoRA: proj/cmix/head квантованы как в REDUCTION, emb НЕТ.

    emb остаётся bf16 не по осторожности: sb6 не умеет выбирать строки
    (коды упакованы блоками вдоль входной оси), поэтому таблицу пришлось
    бы разворачивать целиком на каждом проходе. Так решено в
    rwkv-metal (RwkvqAttachOptions.quantizeEmbedding=false), и здесь
    состав повторяется, иначе мерялась бы не та база.
    """
    cfg = copy.deepcopy(REDUCTION)
    cfg.act_stats_path = ACT_STATS
    cfg.bits.update({g: 16 for g in LORA_GROUPS})
    cfg.bits.update(bits)
    cfg.bits_overrides = {"emb.weight": 16}
    return cfg


def _reduction() -> QuantConfig:
    cfg = copy.deepcopy(REDUCTION)
    cfg.act_stats_path = ACT_STATS
    return cfg


CONFIGS = {
    "bf16": QuantConfig(),
    "qlora_pth": _qlora(),
    "qlora_wav": _qlora(w_lora=6, a_lora=6, v_lora=6),
    "qlora_all": _qlora(w_lora=6, a_lora=6, v_lora=6, g_lora=8),
    "REDUCTION": _reduction(),
}


def weight_diagnostic(sd, cfg):
    """Дешёвая часть: относительная ошибка деквантованных LoRA-тензоров.

    Не решает вопрос (ошибка в весах не переводится в ppl напрямую), но
    показывает, ГДЕ она сидит, и ловит грубые промахи раньше, чем на них
    будет потрачен час ppl-прогонов."""
    print("\nошибка деквантования LoRA-веток (относительная, по Фробениусу):",
          flush=True)
    acc = {}
    for key, w in sd.items():
        if w.dim() < 2:
            continue
        qt = quantize_tensor(key, w, cfg, real_gw=True)
        if qt.bits >= 16 or qt.group not in LORA_GROUPS:
            continue
        w32 = w.float()
        d = _dequantize_one(qt).float()
        num = float((d - w32).pow(2).sum())
        den = float(w32.pow(2).sum()) + 1e-30
        a = acc.setdefault(qt.group, [0.0, 0.0, 0, qt.gw_mode or "rtn"])
        a[0] += num
        a[1] += den
        a[2] += 1
    for g in LORA_GROUPS:
        if g in acc:
            num, den, n, mode = acc[g]
            print(f"  {g:<8} {mode:<5} {n:>4} тензоров   "
                  f"rel {np.sqrt(num / den):.4e}", flush=True)
    return {g: {"rel": float(np.sqrt(v[0] / v[1])), "n": v[2], "mode": v[3]}
            for g, v in acc.items()}


def build(sd, cfg, meta):
    tensors = {key: quantize_tensor(key, w, cfg, real_gw=True)
               for key, w in sd.items()}
    return QuantizedCheckpoint(tensors=tensors, config_repr=repr(cfg),
                               config=cfg, **meta)


def main():
    names = sys.argv[1:] or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        print(f"неизвестные конфиги: {unknown}; есть {list(CONFIGS)}")
        return 1
    if not os.path.exists(ACT_STATS):
        print(f"!! нет {ACT_STATS} -- AW-режимы выродятся и цифры будут не "
              f"про пресет (закон 9 наоборот). Сначала collect_act_stats.py")
        return 1

    blob = torch.load(CORPUS)
    data, langs = blob["tokens"].numpy(), blob["lang"]
    print(f"корпус {data.shape}, "
          f"{ {l: langs.count(l) for l in sorted(set(langs))} }", flush=True)

    print(f"загрузка {os.path.basename(CKPT_PTH)} ...", flush=True)
    sd = torch.load(CKPT_PTH, map_location="cpu", mmap=True)   # закон 13
    from rwkv_quant.formats.writer import detect_meta
    meta = detect_meta(CKPT_PTH, sd)
    print(f"  {len(sd)} тензоров, {meta}", flush=True)

    results = {}
    if os.path.exists(OUT_JSON):
        results = json.load(open(OUT_JSON))

    if "qlora_all" in names:
        results["_weights"] = weight_diagnostic(sd, CONFIGS["qlora_all"])

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        ckpt = build(sd, CONFIGS[name], meta)
        print(f"  квантование {time.time()-t0:.0f}s, "
              f"{checkpoint_mb(ckpt):.1f} МБ", flush=True)
        model = QuantRWKV7(ckpt)
        t0 = time.time()
        per_seq = nll_per_seq(model, data)
        res = ppl_grouped(per_seq, langs)
        print(f"  ppl за {time.time()-t0:.0f}s: "
              + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
        results[name] = {"ppl": res, "size_mb": checkpoint_mb(ckpt)}
        del model, ckpt
        gc.collect()
        mx.clear_cache()
        json.dump(results, open(OUT_JSON, "w"), indent=2)

    report(results)
    print(f"\n-> {OUT_JSON}")
    return 0


def report(results):
    base = results.get("bf16", {}).get("ppl")
    if base is None:
        # на 2.9B bf16 через quantize_tensor не строится: там .clone() =
        # 5.9 ГБ анонимной памяти поверх уже отображённого файла (см.
        # eval_2p9b_one.py). Якорь берётся из ранее опубликованного
        # замера на ТОМ ЖЕ корпусе и харнессе -- на 1.5B они совпали до
        # четвёртого знака, так что подстановка законна.
        anchor = os.environ.get("RWKVQ_BF16_ALL")
        if anchor is None:
            print("\n(bf16 не мерен -- задайте RWKVQ_BF16_ALL для якоря)")
            langs = None
        else:
            print(f"\n(bf16 ALL взят внешним якорем: {anchor})")
            langs = None
        if langs is None:
            a, b = results.get("qlora_pth"), results.get("qlora_all")
            if a and b:
                _answer(a, b, results.get("qlora_wav"), list(a["ppl"]))
            return
    langs = list(base)
    print("\n" + "=" * 72)
    print(f"{'конфиг':<12}" + "".join(f"{l:>15}" for l in langs))
    for name in CONFIGS:
        r = results.get(name)
        if not r:
            continue
        print(f"{name:<12}" + "".join(
            f"{r['ppl'][l]:>8.3f}({100*(r['ppl'][l]-base[l])/base[l]:+.2f}%)"
            for l in langs))

    a, b = results.get("qlora_pth"), results.get("qlora_all")
    if a and b:
        _answer(a, b, results.get("qlora_wav"), langs)


def _answer(a, b, c, langs):
    print("\nСОБСТВЕННО ОТВЕТ -- цена перехода на .rwkvq целиком:")
    for l in langs:
        d = 100 * (b["ppl"][l] - a["ppl"][l]) / a["ppl"][l]
        print(f"  {l:<5} {a['ppl'][l]:8.4f} -> {b['ppl'][l]:8.4f}  {d:+.3f}%")
    if c:
        print("  атрибуция по ALL: w/a/v "
              f"{100*(c['ppl']['ALL']-a['ppl']['ALL'])/a['ppl']['ALL']:+.3f}%, "
              "g_lora "
              f"{100*(b['ppl']['ALL']-c['ppl']['ALL'])/c['ppl']['ALL']:+.3f}%")


if __name__ == "__main__":
    sys.exit(main())
