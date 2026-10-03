"""03-04.10: сводка оценок цепочки cc7 (fe0310/*.json, file_eval_0310: text 36 / code 24 закрытый / code_open 48).
1) таблица плеч по моделям; 2) реплики GPTQ (перестановка порядка окон калибровки): разброс средних KL и dppl между
репликами; эффект автоподбора поверх GPTQ и «измерено / произведение» с усреднением по репликам, интервал --
бутстрэп по окнам (20000, сид 20261004), на 2.9B совместно с ресэмплингом реплик.
    python fe_summary_0310.py <каталог fe0310>"""
import glob, json, math, os, sys
import numpy as np
D = sys.argv[1]; B = 20000
J = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(D, "*.json")))}
KINDS = ("text", "code", "code_open")
def arr(name, f): return np.array(J[name][f], dtype=np.float64)
def idx(name, k): return np.where(np.array(J[name]["kind"]) == k)[0]
def kl(name, k): return arr(name, "kl")[idx(name, k)].mean()
def dppl(name, k):
    i = idx(name, k); return 100 * (math.exp(arr(name, "ce")[i].mean() - arr(name, "ce_ref")[i].mean()) - 1)
LABS = [l for l in ("0p1b", "0p4b", "1p5b", "2p9b", "7p2b", "13b") if any(n.startswith(l + "_") for n in J)]
ARMS = (("REDUCTION", "rtn_red"), ("REDUCTION+GPTQ", "red_gptq"), ("COMPRESSION", "rtn_comp"), ("+автоподбор", "rtn_wprop"),
        ("+GPTQ", "comp_gptq_s0"), ("+оба (умолчание)", "comp_both_s0"))
print("== 1. Плечи (s0). МБ | KL текст / код закр. / код откр. | dppl % текст / код закр. / код откр.")
for l in LABS:
    ref = None
    for title, a in ARMS:
        n = "%s_%s" % (l, a)
        if n not in J: print("%-5s %-18s -- нет" % (l, title)); continue
        if ref is None: ref = arr(n, "ce_ref")
        assert np.abs(arr(n, "ce_ref") - ref).max() < 1e-4, (n, "ce_ref")
        print("%-5s %-18s %8.1f | %.5f / %.5f / %.5f | %+.2f / %+.2f / %+.2f" % ((l, title, J[n]["bytes"] / 1e6) + tuple(kl(n, k) for k in KINDS) + tuple(dppl(n, k) for k in KINDS)))
print()
print("== 2. Реплики GPTQ")
rng = np.random.default_rng(20261004)
for l in LABS:
    both = sorted(n for n in J if n.startswith(l + "_comp_both_s")); gp = sorted(n for n in J if n.startswith(l + "_comp_gptq_s"))
    if len(both) < 2: continue
    print("-- %s: умолчание реплик %d (%s), «только GPTQ» реплик %d" % (l, len(both), ",".join(n.split("_s")[-1] for n in both), len(gp)))
    for grp, title in ((both, "умолчание"), (gp, "только GPTQ")):
        if len(grp) < 2: continue
        for k in KINDS:
            v = np.array([kl(n, k) for n in grp]); p = np.array([dppl(n, k) for n in grp])
            perm = v[1:] if grp[0].endswith("_s0") else v
            # шум по окну: sd реплик / среднее, медиана по окнам
            Wm = np.stack([arr(n, "kl")[idx(n, k)] for n in grp]); cvw = np.median(Wm.std(0, ddof=1) / Wm.mean(0))
            print("   %-12s %-9s KL %s | среднее %.5f, sd %.2f%%, размах %.2f%%, s0 к среднему перестановок %+.2f%% | dppl %s (размах %.2f п.) | шум окна (медиана sd) %.1f%%" % (
                title, k, " ".join("%.5f" % x for x in v), v.mean(), 100 * v.std(ddof=1) / v.mean(), 100 * (v.max() - v.min()) / v.mean(),
                100 * (v[0] / perm.mean() - 1) if grp[0].endswith("_s0") else float("nan"), " ".join("%+.2f" % x for x in p), p.max() - p.min(), 100 * cvw))
    comp, auto = "%s_rtn_comp" % l, "%s_rtn_wprop" % l
    for k in KINDS:
        i = idx(both[0], k); n = len(i)
        Kb = np.stack([arr(x, "kl")[i] for x in both]); Kg = np.stack([arr(x, "kl")[i] for x in gp]) if gp else None
        kc, ka = arr(comp, "kl")[i], arr(auto, "kl")[i]
        S = rng.integers(0, n, size=(B, n))
        # умолчание к пресету: среднее по репликам, окна бутстрэпом, реплики тоже ресэмплируются
        Rb = rng.integers(0, len(both), size=(B, len(both)))
        mb = np.stack([Kb[Rb[b]][:, S[b]].mean() for b in range(B)])
        d = Kb.mean() / kc.mean() - 1; db = mb / kc[S].mean(1) - 1
        per = [Kb[r].mean() / kc.mean() - 1 for r in range(len(both))]
        line = "   %-9s умолчание к пресету %+.1f%% [%+.1f; %+.1f] (по репликам %s)" % (k, 100 * d, *(100 * np.percentile(db, [2.5, 97.5])), " ".join("%+.1f" % (100 * x) for x in per))
        if Kg is not None and len(gp) >= 2:
            Rg = rng.integers(0, len(gp), size=(B, len(gp)))
            mg = np.stack([Kg[Rg[b]][:, S[b]].mean() for b in range(B)])
            e = Kb.mean() / Kg.mean() - 1; eb = mb / mg - 1
            pairs = [Kb[a].mean() / Kg[c].mean() - 1 for a in range(len(both)) for c in range(len(gp))]
            mult = (Kb.mean() / kc.mean()) / ((Kg.mean() / kc.mean()) * (ka.mean() / kc.mean()))
            multb = (mb / kc[S].mean(1)) / ((mg / kc[S].mean(1)) * (ka[S].mean(1) / kc[S].mean(1)))
            line += "\n             автоподбор поверх GPTQ %+.1f%% [%+.1f; %+.1f]; по парам реплик от %+.1f до %+.1f; измерено / произведение %.3f [%.3f; %.3f]" % (
                100 * e, *(100 * np.percentile(eb, [2.5, 97.5])), 100 * min(pairs), 100 * max(pairs), mult, *np.percentile(multb, [2.5, 97.5]))
        elif Kg is not None:
            e = [Kb[r].mean() / Kg[0].mean() - 1 for r in range(len(both))]
            mult = [(Kb[r].mean() / kc.mean()) / ((Kg[0].mean() / kc.mean()) * (ka.mean() / kc.mean())) for r in range(len(both))]
            line += "\n             автоподбор поверх GPTQ (одна реплика GPTQ s0) по репликам умолчания %s; измерено / произведение %s" % (
                " ".join("%+.1f" % (100 * x) for x in e), " ".join("%.3f" % x for x in mult))
        print(line)
