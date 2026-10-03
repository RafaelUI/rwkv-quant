"""03.10: парный бутстрэп по окнам (20000 повторов, сид 20261003) для разделов 11-13 статьи.
Данные -- JSON серверных прогонов 30.09-01.10 (matrix_*, gptq_*_c2w600[_wprop], gptq_red_*), окна одни и те же
(36 текст + 24 код), проверяется равенством kind/langs и ce.ref между файлами одной модели.
Относительное: KL -- отношение средних (b / a - 1); ppl -- exp(mean CE b - mean CE a) - 1; окна ресэмплируются ВМЕСТЕ.
Абсолютное: mean KL и exp(mean CE - mean CE ref) - 1 с интервалом по окнам.
Интервал НЕ включает разброс GPTQ между реализациями (~+-5% KL, раздел 12).
Запуск из каталога с JSON:  python boot_matrix_0310.py [метки...]"""
import json, math, sys
import numpy as np
B = 20000
LABS = sys.argv[1:] or ["0p1b", "0p4b", "1p5b", "2p9b", "7p2b", "13b"]
SL = (("текст", "text", None), ("en", None, "en"), ("ru", None, "ru"), ("sr", None, "sr"), ("код", "code", None))

def load(lab):
    m = json.load(open("matrix_%s.json" % lab))
    g = json.load(open("gptq_%s_c2w600.json" % lab))
    w = json.load(open("gptq_%s_c2w600_wprop.json" % lab))
    r = json.load(open("gptq_red_%s.json" % lab))
    for d, nm in ((g, "gptq"), (w, "gptq_wprop"), (r, "gptq_red")):
        assert d["kind"] == m["kind"] and d["langs"] == m["langs"], (lab, nm, "окна")
        dref = np.abs(np.array(d["ce"]["ref"]) - np.array(m["ce"]["ref"])).max()
        assert dref < 1e-5, (lab, nm, "ce.ref", dref)
    assert g["overrides"] is False and w["overrides"] is True and g["ncal"] == w["ncal"] == r["ncal"] == 600
    A = lambda x: np.array(x, dtype=np.float64)
    # 7.2B / 13.3B: в matrix_* только red; comp и auto -- RTN-плечи GPTQ-прогонов (тот же прибор, те же окна)
    mk = lambda t, name, alt: A(m[t][name]) if name in m[t] else A(alt[t]["rtn"])
    kl = {"red": A(m["kl"]["red"]), "red_gptq": A(r["kl"]["gptq"]), "comp": mk("kl", "comp", g), "auto": mk("kl", "wprop", w),
          "gptq": A(g["kl"]["gptq"]), "both": A(w["kl"]["gptq"])}
    ce = {"ref": A(m["ce"]["ref"]), "red": A(m["ce"]["red"]), "red_gptq": A(r["ce"]["gptq"]), "comp": mk("ce", "comp", g),
          "auto": mk("ce", "wprop", w), "gptq": A(g["ce"]["gptq"]), "both": A(w["ce"]["gptq"])}
    # сверка прибора: RTN-плечи GPTQ-прогонов против отдельного замера
    chk = {"comp~gptq.rtn": np.abs(A(g["kl"]["rtn"]) - kl["comp"]).max(), "auto~wprop.rtn": np.abs(A(w["kl"]["rtn"]) - kl["auto"]).max(),
           "red~red.rtn": np.abs(A(r["kl"]["rtn"]) - kl["red"]).max()}
    return np.array(m["kind"]), np.array(m["langs"]), kl, ce, chk

def mask(kind, lang, k, l):
    return np.where(kind == k)[0] if k else np.where((lang == l) & (kind == "text"))[0]

PAIRS = (("автоподбор к пресету (разд. 11)", "comp", "auto"), ("GPTQ к RTN (разд. 12)", "comp", "gptq"),
         ("GPTQ поверх автоподбора", "auto", "both"), ("автоподбор поверх GPTQ", "gptq", "both"),
         ("умолчание к пресету (разд. 12-13)", "comp", "both"), ("GPTQ в REDUCTION (разд. 14, сверка)", "red", "red_gptq"))
ARMS = ("red", "red_gptq", "comp", "auto", "gptq", "both")
for lab in LABS:
    rng = np.random.default_rng(20261003)
    kind, lang, kl, ce, chk = load(lab)
    print("== %s: %d текст + %d код; сверка RTN-плеч max|d KL|: %s" % (lab, (kind == "text").sum(), (kind == "code").sum(),
          ", ".join("%s %.1e" % kv for kv in chk.items())))
    S = {}
    for name, k, l in SL:
        idx = mask(kind, lang, k, l); S[name] = idx[rng.integers(0, len(idx), size=(B, len(idx)))]
    print("-- абсолютные (разд. 13): KL [95%] | dppl % [95%]")
    for a in ARMS:
        out = []
        for name in ("текст", "код"):
            idx = mask(kind, lang, *[(k, l) for n, k, l in SL if n == name][0]); s = S[name]
            klb = kl[a][s].mean(1); dp = np.exp(ce[a][s].mean(1) - ce["ref"][s].mean(1)) - 1
            out.append("%s KL %.5f [%.5f; %.5f] dppl %+.2f [%+.2f; %+.2f]" % (name, kl[a][idx].mean(), *np.percentile(klb, [2.5, 97.5]),
                       100 * (math.exp(ce[a][idx].mean() - ce["ref"][idx].mean()) - 1), *(100 * np.percentile(dp, [2.5, 97.5]))))
        print("  %-9s %s" % (a, " | ".join(out)))
    for title, a, b in PAIRS:
        print("-- %s: %s -> %s" % (title, a, b))
        for name, k, l in SL:
            idx = mask(kind, lang, k, l); s = S[name]; n = len(idx)
            r = kl[b][idx].mean() / kl[a][idx].mean() - 1
            rb = kl[b][s].mean(1) / kl[a][s].mean(1) - 1
            dp = math.exp(ce[b][idx].mean() - ce[a][idx].mean()) - 1
            dpb = np.exp(ce[b][s].mean(1) - ce[a][s].mean(1)) - 1
            print("  %-5s KL %+6.1f%% [%+6.1f; %+6.1f] хуже %2d/%d | ppl %+6.2f%% [%+6.2f; %+6.2f] P(хуже) %.3f" % (
                name, 100 * r, *(100 * np.percentile(rb, [2.5, 97.5])), int((kl[b][idx] > kl[a][idx]).sum()), n,
                100 * dp, *(100 * np.percentile(dpb, [2.5, 97.5])), (dpb > 0).mean()))
    # мультипликативность: (both/comp) против (gptq/comp)*(auto/comp), интервал отношения
    for name in ("текст", "код"):
        idx = mask(kind, lang, *[(k, l) for n, k, l in SL if n == name][0]); s = S[name]
        f = lambda I, ax=None: (kl["both"][I].mean(ax) / kl["comp"][I].mean(ax)) / ((kl["gptq"][I].mean(ax) / kl["comp"][I].mean(ax)) * (kl["auto"][I].mean(ax) / kl["comp"][I].mean(ax)))
        fb = f(s, 1)
        print("-- сложение методов, %s: измерено/произведение = %.3f [%.3f; %.3f] (1 = независимы)" % (name, f(idx), *np.percentile(fb, [2.5, 97.5])))
