"""Перемер композитных конфигов на РЕАЛЬНОМ пути (`real_gw=True`).

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ. Все числа по Q6_K-раскладке (12-13.08) сняты
fake-путём: веса прогонялись через квантование и обратно, а считал их
обычный bf16-forward. Реальный путь -- это упакованные байты и Metal-GEMV
поверх них, и он даёт другое число: по прежним замерам поправка
0.07-0.08 п.п., одинаковая на обоих масштабах. Пока упаковщика sym не
было, померить её на sym было нечем; теперь есть.

КОНФИГИ ИМПОРТИРУЮТСЯ ИЗ ablate_sym_composite, а не переписываются здесь.
Это не экономия строк: два списка конфигов рано или поздно разъезжаются, и
тогда «поправка fake->real» окажется разницей между двумя РАЗНЫМИ
схемами, а не между двумя путями одной. Здесь сравниваются ровно те же
объекты QuantConfig.

МБ СЧИТАЮТСЯ ПО БАЙТАМ БУФЕРОВ, а не формулой: в этом и смысл реального
пути. Поля перечисляются из writer.TENSOR_FIELDS -- список один, и новое
поле (gw_qs) не потеряется молча, как это уже случалось с asym gw64.

Один конфиг на процесс (законы 11 и 13). Память -- только системными
командами (закон 11/22): vm.swapusage и memory_pressure до и после.

    RWKVQ_CKPT=... RWKVQ_ACT_STATS=... python tests/eval_real_sym.py <конфиг>
    python tests/eval_real_sym.py --report
"""
import gc
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.schema import (QuantizedCheckpoint,  # noqa: E402
                                       QuantizedTensor)
from rwkv_quant.formats.writer import TENSOR_FIELDS, quantize_tensor  # noqa: E402

import ablate_sym_composite as comp  # noqa: E402

CKPT = comp.CKPT
CORPUS = comp.CORPUS
NSEQ, SEQLEN = comp.NSEQ, comp.SEQLEN
OUT = os.environ.get("RWKVQ_REAL_OUT", "/tmp/sym_real.json")


def mem(tag):
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True).stdout.strip()
    mp = subprocess.run(["memory_pressure", "-Q"],
                        capture_output=True, text=True).stdout
    free = next((l.split(":")[1].strip() for l in mp.splitlines()
                 if "free percentage" in l), "?")
    print(f"  [mem/{tag}] {sw} | свободно {free}", flush=True)


def tensor_bytes(qt):
    return sum(getattr(qt, f).numel() * getattr(qt, f).element_size()
               for f in TENSOR_FIELDS if getattr(qt, f, None) is not None)


def detect_meta(sd):
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    emb = sd["emb.weight"]
    r_k = next(v for k, v in sd.items() if k.endswith("r_k"))
    return dict(naming="world", n_layer=n_layer, n_embd=int(emb.shape[1]),
                vocab_size=int(emb.shape[0]), head_size=int(r_k.shape[-1]))


def nll_per_seq(model, data):
    """ПО ОДНОЙ последовательности: батч на 2.9B не помещается, а разбивка
    по языкам всё равно нужна попоследовательностная."""
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


def ppl(pairs):
    s, n = sum(x[0] for x in pairs), sum(x[1] for x in pairs)
    mean = s / n
    # закон 21: ppl = 1.0000 -- это упавший command buffer, а не результат
    if not (mean > 1e-6) or mean != mean:
        raise RuntimeError(f"ppl вырождена (NLL={mean!r}): смотрите stderr "
                           f"на 'Insufficient Memory'")
    return float(np.exp(mean))


def report():
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    rows = doc.get("rows", {})
    fake_path = os.environ.get("RWKVQ_OUT", "/tmp/sym_composite.json")
    fake = (json.load(open(fake_path)).get("rows", {})
            if os.path.exists(fake_path) else {})
    bf = rows.get("bf16", {}).get("ppl")
    fbf = fake.get("bf16", {}).get("ppl")
    langs = sorted(set(doc.get("langs", [])))
    print(f"\nреальный путь, {doc.get('n_pred','?')} предсказаний, "
          f"{os.path.basename(doc.get('ckpt',''))}, bf16 ppl={bf}")
    head = (f"\n{'конфиг':26s} {'МБ файла':>9s} {'ppl':>9s} {'Δ real':>8s} "
            f"{'Δ fake':>8s} {'поправка':>9s}")
    for l in langs:
        head += f" {('Δ ' + l):>8s}"
    print(head)
    for name in comp.CONFIGS:
        r = rows.get(name)
        if not r:
            continue
        d = 100 * (r["ppl"] - bf) / bf if bf else None
        f = fake.get(name)
        df = 100 * (f["ppl"] - fbf) / fbf if (f and fbf) else None
        line = (f"{name:26s} {r.get('mb', 0):9.1f} {r['ppl']:9.4f} "
                f"{d:+7.3f}% " + (f"{df:+7.3f}% {d-df:+8.3f}" if df is not None
                                  else f"{'-':>8s} {'-':>9s}"))
        for l in langs:
            pl = r.get("by_lang", {}).get(l)
            bl = rows.get("bf16", {}).get("by_lang", {}).get(l)
            line += f" {100*(pl-bl)/bl:+7.3f}%" if pl and bl else f" {'-':>8s}"
        print(line)
    print("\nпоправка = Δ на реальном пути минус Δ на fake-пути; по прежним "
          "замерам она 0.07-0.08 п.п. и одинакова на обоих масштабах")


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--report":
        report()
        print(f"\nконфиги: {' '.join(comp.CONFIGS)}")
        return 0
    name = a[0]
    if name not in comp.CONFIGS:
        raise SystemExit(f"неизвестный конфиг {name}; есть: {list(comp.CONFIGS)}")
    cfg = comp.CONFIGS[name]()
    if name != "bf16" and not os.path.exists(comp.ACT):
        raise SystemExit(f"нет статистики {comp.ACT}: AW-режимы выродятся в "
                         f"_search, и замер будет не тот (закон 15)")

    print(f"=== {os.path.basename(CKPT)} / {name} / РЕАЛЬНЫЙ путь ===", flush=True)
    mem("старт")
    blob = torch.load(CORPUS)
    data = blob["tokens"][:NSEQ, :SEQLEN].numpy()
    langs = list(blob["lang"])[:NSEQ]

    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    meta = detect_meta(sd)
    mem("после mmap")

    t0 = time.time()
    if name == "bf16":
        # без .clone(): страницы остаются file-backed, иначе это честные
        # 5.9 ГБ анонимной памяти поверх уже отображённого файла
        tensors = {k: QuantizedTensor(key=k, group="other", bits=16,
                                      shape=tuple(w.shape),
                                      dense=w if w.dtype == torch.bfloat16
                                      else w.to(torch.bfloat16))
                   for k, w in sd.items()}
    else:
        tensors = {}
        for i, (k, w) in enumerate(sd.items()):
            tensors[k] = quantize_tensor(k, w, cfg, real_gw=True)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(sd)}", flush=True)
    mb = sum(tensor_bytes(q) for q in tensors.values()) / 1e6
    print(f"  упаковка {time.time()-t0:.0f}s, {mb:.1f} МБ буферов", flush=True)
    mem("после упаковки")

    model = QuantRWKV7(QuantizedCheckpoint(tensors=tensors, config_repr=repr(cfg),
                                           **meta))
    del tensors, sd
    gc.collect()
    mem("после постройки модели")

    t0 = time.time()
    pairs = nll_per_seq(model, data)
    p = ppl(pairs)
    by_lang = {l: ppl([x for x, ll in zip(pairs, langs) if ll == l])
               for l in sorted(set(langs))}

    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    doc.update(ckpt=CKPT, corpus=CORPUS, act=comp.ACT, langs=langs,
               n_pred=sum(x[1] for x in pairs))
    doc["rows"][name] = {"ppl": p, "mb": mb, "by_lang": by_lang,
                         "seconds": round(time.time() - t0)}
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"{name}: ppl={p:.4f}  {mb:.1f} МБ  [{time.time()-t0:.0f}s]", flush=True)
    print("по языкам: " + "  ".join(f"{k} {v:.4f}" for k, v in by_lang.items()))
    mem("финиш")
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
