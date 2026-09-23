"""Отложенные окна fake-путём (ws_<m>_ho.json от run_heldout_2209.sh): группы -> 30 окон,
ПАРНО к live по окнам, бутстрэп. Для 1.5B/2.9B -- сверка прибора с kl_file (файлы sym*).
    python heldout_joint_2209.py [каталог=~/rwkvq]"""
import json, os, sys
import numpy as np
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/rwkvq")
LANGS = json.load(open(f"{D}/rule_files_a/kl_1p5b_live.json"))["langs"]
rng = np.random.default_rng(0)
for m in ["1p5b", "2p9b", "7p2b", "13b"]:
    f = f"{D}/ws_{m}_ho.json"
    if not os.path.exists(f):
        continue
    J = json.load(open(f))["joint"]; cfgs = {}
    for k, v in J.items():
        name = k.rsplit("_g", 1)[0]
        for w, x in zip(v["win"], v["per_win"]):
            cfgs.setdefault(name, {})[w] = (x, v["add_bytes"])
    if "live" not in cfgs:
        continue
    L = cfgs["live"]
    print("=== %s: live на %d отлож. окнах KL %.5f" % (m, len(L), np.mean([x for x, _ in L.values()])))
    for name, c in cfgs.items():
        if name == "live":
            continue
        ws = sorted(set(c) & set(L))
        if len(ws) < 30:
            print("   %-6s неполно (%d окон)" % (name, len(ws))); continue
        a = np.array([c[w][0] for w in ws]); b = np.array([L[w][0] for w in ws]); d = a - b
        bs = [d[i].sum() / b[i].sum() for i in (rng.integers(0, len(ws), len(ws)) for _ in range(4000))]
        lo, hi = np.percentile(bs, [2.5, 97.5]); lg = np.array([LANGS[w] for w in ws])
        by = " ".join("%s %+.1f%%" % (g, 100 * d[lg == g].sum() / b[lg == g].sum()) for g in ("ru", "en", "sr"))
        dmb = (c[ws[0]][1] - L[ws[0]][1]) / 1e6
        line = "   %-6s %+7.1f МБ  KL %+6.1f%% [%+.1f; %+.1f]  хуже окон %2d/30 | %s" % (
            name, dmb, 100 * d.sum() / b.sum(), 100 * lo, 100 * hi, int((d > 0).sum()), by)
        ff = f"{D}/sym_files/kl_{m}_{name}.json"
        fl = f"{D}/rule_files_a/kl_{m}_live.json"
        if os.path.exists(ff) and os.path.exists(fl):
            X, Y = json.load(open(ff)), json.load(open(fl))
            rf = sum(X["kl"][w] - Y["kl"][w] for w in ws) / sum(Y["kl"][w] for w in ws)
            dev = max(abs(X["kl"][w] - c[w][0]) / X["kl"][w] for w in ws)
            line += " || файл: %+.1f%%, fake vs файл по окну max отн. %.1e" % (100 * rf, dev)
        print(line)
