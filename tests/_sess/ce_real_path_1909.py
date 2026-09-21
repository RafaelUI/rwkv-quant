"""ppl И KL ИЗ ОДНИХ И ТЕХ ЖЕ ЛОГИТОВ на реальном пути (19.09): для файла
.rwkvq считает на 38 окнах eval_corpus_multiling кросс-энтропию на настоящих
следующих токенах (-> ppl) и KL против fp32-эталона; эталон даёт и базовую
ppl. Цель -- проверить, воспроизводится ли эффект калибровки, видный по ppl
(17-18.09, 0.4 п.п.), на ТЕХ ЖЕ файлах, где KL его почти не видит.
Пер-окно CE пишутся в json; разности плеч -- парный бутстрэп по окнам.
    RWKVQ_KL_REF=... python ce_real_path_1909.py <файл|--ref> <имя> <out.json>
    python ce_real_path_1909.py --report <out.json>
"""
import os, sys, json
sys.path.insert(0, "/Users/s/Develop/rwkv-quant"); sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")
import numpy as np
import torch

CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")


def lse(x):
    m = x.max(1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(1, keepdims=True)))[:, 0]


def ce_of(lg, tgt):
    lg = lg.astype(np.float64)
    return lse(lg) - lg[np.arange(len(tgt)), tgt]


def run(path, name, out):
    from ablate_subgroups import kl_stats
    blob = torch.load(CORPUS)
    data = blob["tokens"][:, :512].numpy(); langs = list(blob["lang"])
    ref = np.load(os.environ["RWKVQ_KL_REF"], mmap_mode="r")
    N = ref.shape[0]
    rows = {"ce": [], "kl": [], "langs": langs[:N]}
    if path != "--ref":
        import mlx.core as mx
        from rwkv_quant.formats.reader import load_raw
        from rwkv_quant.backends.metal.quant_model import QuantRWKV7
        model = QuantRWKV7(load_raw(path), fast_ln=(True if os.environ.get("RWKVQ_BENCH_FAST_LN") == "1" else None))
    for i in range(N):
        tgt = data[i, 1:512]
        if path == "--ref":
            got = np.asarray(ref[i])
        else:
            lg = model(mx.array(data[i:i + 1, :-1])).astype(mx.float32); mx.eval(lg)
            got = np.array(lg)[0]; del lg
            k, _ = kl_stats(np.asarray(ref[i]), got)
            rows["kl"].append(float(k.mean()))
        rows["ce"].append(float(ce_of(got, tgt).mean()))
    doc = json.load(open(out)) if os.path.exists(out) else {}
    doc[name] = rows
    json.dump(doc, open(out, "w"))
    print("%s: ppl %.4f  KL %s" % (name, float(np.exp(np.mean(rows["ce"]))),
          "%.6f" % np.mean(rows["kl"]) if rows["kl"] else "-"), flush=True)


def boot(d, it=20000, seed=0):
    r = np.random.default_rng(seed); idx = r.integers(0, len(d), (it, len(d)))
    m = d[idx].mean(1); return np.percentile(m, 2.5), np.percentile(m, 97.5)


def report(out):
    doc = json.load(open(out))
    base = np.array(doc["ref"]["ce"]); L = np.array(doc["ref"]["langs"])
    print("плечо        Δppl ALL   en      ru      sr      KL")
    for n, r in doc.items():
        if n == "ref":
            continue
        ce = np.array(r["ce"])
        dp = lambda m: 100 * (np.exp(ce[m].mean()) / np.exp(base[m].mean()) - 1)
        print("%-10s %+7.3f%% %+6.2f %+6.2f %+6.2f  %.6f" % (n, dp(np.ones(len(ce), bool)), dp(L == "en"), dp(L == "ru"), dp(L == "sr"), np.mean(r["kl"])))
    names = [n for n in doc if n != "ref"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = np.array(doc[names[i]]["ce"]), np.array(doc[names[j]]["ce"])
            d = a - b; lo, hi = boot(d)
            print("  %s − %s: ΔCE %+.5f нат/ток (≈ %+.2f п.п. ppl)  95%% CI [%+.5f; %+.5f]  %s" % (
                names[i], names[j], d.mean(), 100 * d.mean(), lo, hi, "ЗНАЧИМО" if lo * hi > 0 else "в шуме"))


if sys.argv[1] == "--report":
    report(sys.argv[2])
else:
    run(sys.argv[1], sys.argv[2], sys.argv[3])
