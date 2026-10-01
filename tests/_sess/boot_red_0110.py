"""01.10: парный бутстрэп по окнам (20000 повторов, сид 20261001) для GPTQ в REDUCTION против RTN-REDUCTION.
KL -- отношение средних KL (gptq / rtn - 1); ppl -- exp(mean CE gptq - mean CE rtn) - 1. Окна ресэмплируются ВМЕСТЕ для обоих плеч.
Запуск из каталога с gptq_red_<метка>.json:  python boot_red_0110.py 2p9b 7p2b ..."""
import json, math, sys
import numpy as np
rng = np.random.default_rng(20261001)
B = 20000
for lab in sys.argv[1:]:
    d = json.load(open("gptq_red_%s.json" % lab))
    kind = np.array(d["kind"]); lang = np.array(d["langs"])
    kr, kg = np.array(d["kl"]["rtn"]), np.array(d["kl"]["gptq"])
    cr, cg = np.array(d["ce"]["rtn"]), np.array(d["ce"]["gptq"])
    print("== %s (%d окон текста, %d кода)" % (lab, (kind == "text").sum(), (kind == "code").sum()))
    for name, m in (("текст", kind == "text"), ("en", lang == "en"), ("ru", lang == "ru"), ("sr", lang == "sr"), ("код", kind == "code")):
        idx = np.where(m)[0]; n = len(idx)
        S = idx[rng.integers(0, n, size=(B, n))]
        kl = kg[idx].mean() / kr[idx].mean() - 1
        klb = kg[S].mean(1) / kr[S].mean(1) - 1
        dp = math.exp(cg[idx].mean() - cr[idx].mean()) - 1
        dpb = np.exp(cg[S].mean(1) - cr[S].mean(1)) - 1
        lo, hi = np.percentile(klb, [2.5, 97.5]); plo, phi = np.percentile(dpb, [2.5, 97.5])
        worse = int((kg[idx] > kr[idx]).sum()); cworse = int((cg[idx] > cr[idx]).sum())
        print("  %-5s KL %+6.1f%% [%+6.1f; %+6.1f]  окон хуже %2d/%d | ppl %+7.3f%% [%+7.3f; %+7.3f]  P(ppl хуже) %.3f  окон хуже по CE %2d/%d" % (
            name, 100 * kl, 100 * lo, 100 * hi, worse, n, 100 * dp, 100 * plo, 100 * phi, (dpb > 0).mean(), cworse, n))
