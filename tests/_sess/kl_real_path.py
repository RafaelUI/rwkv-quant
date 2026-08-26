# -*- coding: utf-8 -*-
"""KL(bf16 ‖ квант) НА РЕАЛЬНОМ ПУТИ -- инструмента с таким разрешением у нас не было.

ЗАЧЕМ. Весь KL в проекте (ablate_subgroups) считается на FAKE-пути: веса
прогоняются через квантование и обратно, а считает их bf16-модель в торче.
Такой инструмент СЛЕП по устройству ко всему, что живёт только в
MLX-пути -- к fp16-контейнеру плотных тензоров, к крюку fp32->bf16->fp16,
к emb-gather. А ppl на реальном пути имеет полосу +-0.05 п.п. при
ожидаемых эффектах в сотые доли.

Здесь логиты считает НАСТОЯЩАЯ QuantRWKV7 из готового .rwkvq, а эталон
берётся тот же самый, что у fake-пути (/tmp/kl_ref_*.npy), поэтому числа
двух путей сравнимы между собой.

Одно плечо на процесс (законы 2 и 13): на 2.9B две модели в одном
процессе машину положат. Пары по последовательностям сводятся в --report,
разности бутстрэпятся ПАРНО (закон 35).

    RWKVQ_KL_REF=... python tests/_sess/kl_real_path.py <файл.rwkvq> <имя плеча> [--nobf16]
    python tests/_sess/kl_real_path.py --report
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch

OUT = os.environ.get("RWKVQ_KLREAL_OUT", "/tmp/kl_real_path.json")
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
NSEQ = int(os.environ.get("RWKVQ_KL_NSEQ", 8))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))


def patch_nobf16():
    """Снять ЛИШНЕЕ округление: деквант отдаёт fp32, а в fp16 его кладёт
    dequantize_banded. Копия _dequantize_gw_sym с одной убранной строкой."""
    import rwkv_quant.formats.reader as R

    def deq(qt):
        OUTd, IN = qt.shape
        gs, NB = qt.gw_gs, IN // qt.gw_gs
        if qt.codes is not None:
            q = qt.codes.to(torch.float32)
        else:
            q = R.unpack_nib_block(qt.codes_packed, gs).to(torch.int16)
            if qt.gw_qh is not None:
                q = q + R.unpack_bitplane(qt.gw_qh, IN).to(torch.int16) * 16
            if qt.gw_qh2 is not None:
                q = q + R.unpack_bitplane(qt.gw_qh2, IN).to(torch.int16) * 32
            q = (q - 32).to(torch.float32)
        d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)
        scale = (qt.gw_qs.float() * d).half().float()
        if NB * gs == IN:
            q = q.view(OUTd, NB, gs)
            q.mul_(scale[..., None])
            return q.view(OUTd, IN)
        return q * scale.repeat_interleave(gs, dim=1)

    R._dequantize_gw_sym = deq


def run(path, name, nobf16):
    import mlx.core as mx
    from rwkv_quant.formats.reader import load_raw
    from rwkv_quant.backends.metal.quant_model import QuantRWKV7
    from ablate_subgroups import kl_stats, boot_ci, REF

    ref_path = os.environ.get("RWKVQ_KL_REF", REF)
    if not os.path.exists(ref_path):
        raise SystemExit("нет эталона %s" % ref_path)
    if nobf16:
        patch_nobf16()

    blob = torch.load(CORPUS)
    data = blob["tokens"][:NSEQ, :SEQLEN].numpy()
    langs = list(blob["lang"])[:NSEQ]
    ref = np.load(ref_path, mmap_mode="r")
    print("=== %s / %s / эталон %s ===" % (os.path.basename(path), name,
                                           os.path.basename(ref_path)), flush=True)

    if path == "bf16":
        # КОНТРОЛЬ БЕЗ КВАНТОВАНИЯ: та же MLX-модель, но веса плотные bf16.
        # Отвечает на вопрос, чей разрыв с fake-путём -- квантования или
        # самого пути. Без .clone(): страницы остаются file-backed.
        from rwkv_quant.formats.schema import (QuantizedCheckpoint,
                                               QuantizedTensor)
        ckpt_path = os.environ["RWKVQ_CKPT"]
        sd = torch.load(ckpt_path, map_location="cpu", mmap=True)
        n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        r_k = next(v for k, v in sd.items() if k.endswith("r_k"))
        meta = dict(naming="world", n_layer=n_layer,
                    n_embd=int(sd["emb.weight"].shape[1]),
                    vocab_size=int(sd["emb.weight"].shape[0]),
                    head_size=int(r_k.shape[-1]))
        tensors = {k: QuantizedTensor(key=k, group="other", bits=16,
                                      shape=tuple(w.shape),
                                      dense=w if w.dtype == torch.bfloat16
                                      else w.to(torch.bfloat16))
                   for k, w in sd.items()}
        model = QuantRWKV7(QuantizedCheckpoint(tensors=tensors,
                                               config_repr="bf16", **meta))
        del tensors, sd
    else:
        model = QuantRWKV7(load_raw(path))

    # КОНТРОЛЬ ВКЛЮЧЕНИЯ, структурный: плечи различаются только числами
    # внутри плотных тензоров, поэтому «патч не применился» и «эффекта нет»
    # по выходу неразличимы. Отпечаток таблицы emb ОБЯЗАН разойтись.
    # Сумма модулей ВСЕЙ таблицы в fp16 даёт inf (переполнение), и тогда
    # два разных плеча дают inf == inf -- контроль молча вырождается в
    # "плечи совпали". Считаем по срезу и В fp32.
    emb = model.emb_weight if not hasattr(model.emb_weight, "codes") else None
    fp = (float(mx.sum(mx.abs(emb[:64].astype(mx.float32))).item())
          if emb is not None else None)
    if fp is not None and not (fp == fp and abs(fp) != float("inf")):
        raise RuntimeError("отпечаток emb вырожден (%r): контроль включения "
                           "не работает, замер недействителен" % fp)
    print("  отпечаток emb: %r" % (fp,), flush=True)

    per_seq, tops = [], []
    for i in range(NSEQ):
        lg = model(mx.array(data[i:i + 1, :-1])).astype(mx.float32)
        mx.eval(lg)
        got = np.array(lg)[0]
        del lg
        if not np.isfinite(got).all() or float(np.abs(got).max()) == 0.0:
            raise RuntimeError("логиты %d вырождены (закон 21)" % i)
        k, t = kl_stats(np.asarray(ref[i]), got)
        per_seq.append(float(k.mean())); tops.append(float(t.mean()))
        print("  %d/%d  KL=%.6f" % (i + 1, NSEQ, per_seq[-1]), flush=True)
        del got

    kl = float(np.mean(per_seq))
    lo, hi = boot_ci(per_seq)
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    doc["ref"] = ref_path; doc["langs"] = langs
    doc["rows"][name] = {"kl": kl, "ci": [lo, hi], "per_seq": per_seq,
                         "top1": float(np.mean(tops)), "file": path,
                         "emb_fp": fp, "nobf16": bool(nobf16)}
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("KL = %.6f нат/токен  95%% CI [%.6f; %.6f]   top-1 %.3f%%"
          % (kl, lo, hi, 100 * np.mean(tops)), flush=True)


def report():
    from ablate_subgroups import boot_ci
    doc = json.load(open(OUT))
    rows = doc["rows"]
    print("\nреальный путь, эталон %s" % os.path.basename(doc["ref"]))
    print("%-22s %10s %26s %9s" % ("плечо", "KL", "95% CI", "top-1"))
    for n, r in rows.items():
        print("%-22s %10.6f  [%.6f; %.6f] %8.3f%%"
              % (n, r["kl"], r["ci"][0], r["ci"][1], 100 * r["top1"]))
    names = list(rows)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = rows[names[i]], rows[names[j]]
            if a.get("emb_fp") is not None and a["emb_fp"] == b.get("emb_fp"):
                print("\n! ОТПЕЧАТКИ emb СОВПАЛИ у %s и %s -- плечи не различаются, "
                      "сравнивать нечего" % (names[i], names[j]))
                continue
            d = np.array(a["per_seq"]) - np.array(b["per_seq"])
            lo, hi = boot_ci(d)
            sig = "ЗНАЧИМО" if lo * hi > 0 else "в шуме"
            print("\n%s − %s: %+.6f нат/ток (%+.2f%%)  95%% CI [%+.6f; %+.6f]  %s"
                  % (names[i], names[j], d.mean(), 100 * d.mean() / b["kl"],
                     lo, hi, sig))


a = sys.argv[1:]
if not a or a[0] == "--report":
    report()
else:
    run(a[0], a[1], "--nobf16" in a)
