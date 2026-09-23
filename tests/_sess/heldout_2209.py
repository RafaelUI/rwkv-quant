"""KL САМИХ ФАЙЛОВ на 30 ОТЛОЖЕННЫХ окнах (вне окон выбора), ПАРНО к live по окнам.
    python heldout_2209.py <kl_live.json> <kl_x.json> ...   (формат kl_file_2209.py)"""
import json, sys
import numpy as np
L = json.load(open(sys.argv[1])); sel = set(L["sel"])
ho = [i for i in range(len(L["kl"])) if i not in sel]
kl0 = np.array(L["kl"]); langs = np.array(L["langs"])
print("live %s: KL все %.5f, отлож. %.5f (%d окон), top-1 %.2f%%"
      % (L["file"].split("/")[-1], kl0.mean(), kl0[ho].mean(), len(ho), 100 * np.mean(L["top1"])))
rng = np.random.default_rng(0)
for f in sys.argv[2:]:
    X = json.load(open(f)); k = np.array(X["kl"])
    assert X["sel"] == L["sel"] and len(k) == len(kl0)
    d = (k[ho] - kl0[ho]); rel = d.sum() / kl0[ho].sum()
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(ho), len(ho)); bs.append(d[i].sum() / kl0[ho][i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    by = " ".join("%s %+.1f%%" % (g, 100 * d[langs[ho] == g].sum() / kl0[ho][langs[ho] == g].sum()) for g in ("ru", "en", "sr"))
    print("%-24s %8.2f МБ (%+6.2f)  отлож. KL %.5f  %+6.1f%% [%+.1f; %+.1f]  хуже окон %2d/%d  top-1 %.2f%%  | %s"
          % (X["file"].split("/")[-1], X["size"] / 1e6, (X["size"] - L["size"]) / 1e6, k[ho].mean(), 100 * rel,
             100 * lo, 100 * hi, int((d > 0).sum()), len(ho), 100 * np.mean(X["top1"]), by))
