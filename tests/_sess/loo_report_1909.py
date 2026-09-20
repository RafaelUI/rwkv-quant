"""Сводка leave-one-out COMPRESSION по масштабам: вклад группы = KL(base) -
KL(без группы), парный бутстрэп по окнам; доля от KL(base); то же для ΔCE."""
import json, sys
import numpy as np


def boot(d, it=20000, seed=0):
    r = np.random.default_rng(seed); idx = r.integers(0, len(d), (it, len(d)))
    m = d[idx].mean(1); return np.percentile(m, 2.5), np.percentile(m, 97.5)


import os
# RWKVQ_LOO_TPL (20.09): шаблон json, напр. .../loo_reduction_%s_2009.json
TPL = os.environ.get("RWKVQ_LOO_TPL", "/Users/s/Develop/WKV-kvant/loo_compression_%s_1909.json")
docs = {s: json.load(open(TPL % s)) for s in sys.argv[1:]}
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

# ПЛЕЧИ ПРАВКИ (20.09): не вклад группы, а пресет с правкой. Выигрыш
# = base - плечо (KL и ppl), парный бутстрэп; ppl по языкам; байты.
def ppl(ce, ref, m=None):
    ce, ref = np.array(ce), np.array(ref)
    if m is not None:
        ce, ref = ce[m], ref[m]
    return 100 * (np.exp(ce.mean()) / np.exp(ref.mean()) - 1)

for s, d in docs.items():
    for a in ["o5", "o6", "o4s"] + sorted(k for k in d if k.startswith("oL")):
        if a not in d or "base" not in d:
            continue
        b, x = d["base"], d[a]
        dd = np.array(b["kl"]) - np.array(x["kl"]); lo, hi = boot(dd)
        dc = np.array(b["ce"]) - np.array(x["ce"]); clo, chi = boot(dc)
        L = np.array(b["langs"])
        print("%s %s: KL %.6f -> %.6f, выигрыш %.5f [%.5f; %.5f] (%.1f%% KL); "
              "ppl %+.3f%% -> %+.3f%%, ΔCE %.5f [%.5f; %.5f]" % (
                  s, a, np.mean(b["kl"]), np.mean(x["kl"]), dd.mean(), lo, hi,
                  100 * dd.mean() / np.mean(b["kl"]), ppl(b["ce"], b["ce_ref"]),
                  ppl(x["ce"], x["ce_ref"]), dc.mean(), clo, chi))
        print("   по языкам base -> %s: " % a + "  ".join(
            "%s %+.2f -> %+.2f" % (l, ppl(b["ce"], b["ce_ref"], L == l), ppl(x["ce"], x["ce_ref"], L == l))
            for l in ("en", "ru", "sr")))
        if "nbytes" in b and "nbytes" in x:
            print("   байты: %.1f -> %.1f МБ (+%.2f МБ, +%.2f%%)" % (
                b["nbytes"] / 1e6, x["nbytes"] / 1e6, (x["nbytes"] - b["nbytes"]) / 1e6,
                100 * (x["nbytes"] - b["nbytes"]) / b["nbytes"]))
