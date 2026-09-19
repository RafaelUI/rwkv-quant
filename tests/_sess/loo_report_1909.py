"""Сводка leave-one-out COMPRESSION по масштабам: вклад группы = KL(base) -
KL(без группы), парный бутстрэп по окнам; доля от KL(base); то же для ΔCE."""
import json, sys
import numpy as np


def boot(d, it=20000, seed=0):
    r = np.random.default_rng(seed); idx = r.integers(0, len(d), (it, len(d)))
    m = d[idx].mean(1); return np.percentile(m, 2.5), np.percentile(m, 97.5)


docs = {s: json.load(open("/Users/s/Develop/WKV-kvant/loo_compression_%s_1909.json" % s)) for s in sys.argv[1:]}
arms = ["r", "k", "v", "o", "ffn_k", "ffn_v", "head", "emb", "lora"]
print("вклад группы в KL (base - без неё), нат/ток; доля от KL base; 95% CI парно")
hdr = "группа  " + "".join("| %-44s" % s for s in docs)
print(hdr)
for a in arms:
    line = "%-7s " % a
    for s, d in docs.items():
        if a not in d or "base" not in d:
            line += "| %-44s" % "-"; continue
        b, x = np.array(d["base"]["kl"]), np.array(d[a]["kl"])
        dd = b - x; lo, hi = boot(dd)
        line += "| %.5f (%4.1f%%) [%.5f; %.5f]    " % (dd.mean(), 100 * dd.mean() / b.mean(), lo, hi)
    print(line)
for s, d in docs.items():
    if "base" not in d:
        continue
    b = d["base"]; L = np.array(b["langs"])
    tot = sum(np.mean(np.array(b["kl"]) - np.array(d[a]["kl"])) for a in arms if a in d)
    print("%s: KL base %.6f, сумма вкладов %.6f (%.0f%%), Δppl base %+.3f%%" % (
        s, np.mean(b["kl"]), tot, 100 * tot / np.mean(b["kl"]),
        100 * (np.exp(np.mean(b["ce"])) / np.exp(np.mean(b["ce_ref"])) - 1)))
    print("  вклад в Δppl, п.п. (base - без группы):  " + "  ".join(
        "%s %+.2f" % (a, 100 * (np.exp(np.mean(b["ce"])) - np.exp(np.mean(d[a]["ce"]))) / np.exp(np.mean(b["ce_ref"])))
        for a in arms if a in d))
