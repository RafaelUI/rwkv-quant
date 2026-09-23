"""ДИЗАЙН ПРАВИЛА И ЦЕНА measure (22.09 ночь). Вход -- полная истина: ws_<m>_ladder_full.json,
ws_<m>_down.json, ws_<m>_joint.json (live).
  A  вниз: сколько байт при e<5 дают шаги на 3 бита против шагов 5->4 (реализуемых сейчас)
  B  СИММЕТРИЧНОЕ ПРАВИЛО tau: вверх -- жадный выбор e>=tau; вниз -- РЕАЛИЗУЕМЫЕ шаги
     (b_to>=4) с e_down<tau у неподнятых плеч. Сетка tau -> байты RAM и KL к live
     (предсказание по одиночным). Пишет sym_<m>_tau<t>.json (ovr поверх live).
  C  дешёвые measure на полной истине (вверх, tau=5): уровни по закону r (медиана r
     по ДРУГИМ масштабам) от истинной базы (V1) или от истинных base и base+1 (V2);
     оценка выбранного -- по истинной лестнице.
    python autopick_design_2209.py [каталог=~/rwkvq]"""
import json, os, sys
import numpy as np
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/rwkvq")
START = {"emb.weight": 6, "blocks.0.att.output.weight": 16}
nb = lambda p, b: 2 * p if b >= 16 else p * (b + 0.5) / 8   # noqa: E731
SC = [m for m in ["0p1b", "0p4b", "1p5b", "2p9b", "7p2b", "13b"]
      if os.path.exists(f"{D}/ws_{m}_ladder_full.json") and os.path.exists(f"{D}/ws_{m}_down.json")]


def typ(a):
    if a in ("emb.weight", "head.weight"):
        return a.split(".")[0]
    t = a.split(".", 2)[2].rsplit(".", 1)[0]
    return "proj" if t.startswith("att.") else "cmix." + t.split(".")[1]


def greedy(lv0, KL_ALL, FILE0, TAU):
    lv, cur, FILE = {}, {}, FILE0
    for a, lev in lv0.items():
        bs = [b for b, _, _ in lev]; cur[a] = bs.index(START.get(a, bs[0])); lv[a] = lev
        FILE += lev[cur[a]][2] - lev[0][2]
    st = dict(cur)
    while True:
        best = None
        for a, lev in lv.items():
            b0, k0, y0 = lev[cur[a]]
            for j in range(cur[a] + 1, len(lev)):
                b1, k1, y1 = lev[j]; db, dkl = y1 - y0, k0 - k1
                e = (dkl / KL_ALL) / (db / FILE) if db > 0 else 0
                if best is None or e > best[0]:
                    best = (e, a, j)
        if best is None or best[0] < TAU:
            break
        cur[best[1]] = best[2]
    return {a: lv[a][cur[a]][0] for a in lv if cur[a] != st[a]}, FILE


def evalT(ovr, T, L):
    d = y = 0.0
    for a, bf in ovr.items():
        s = START.get(a, L[a]["base_bits"]); kk = {b: k for b, k, _ in T[a]}
        d += kk[s] - kk[bf]
        if a != "emb.weight":
            y += nb(L[a]["params"], bf) - nb(L[a]["params"], s)
    return d, y


DATA, RS = {}, {}
for m in SC:
    lad = json.load(open(f"{D}/ws_{m}_ladder_full.json")); L = lad["ladder"]
    DN = json.load(open(f"{D}/ws_{m}_down.json"))["down"]
    KL_ALL = json.load(open(f"{D}/ws_{m}_joint.json"))["joint"]["live"]["kl"]
    T = {a: sorted((int(b), k, nb(e["params"], int(b))) for b, k in e["kl"].items()) for a, e in L.items()}
    DATA[m] = (lad["_meta"], L, T, DN, KL_ALL)
    for a, e in L.items():
        ks = {int(b): k for b, k in e["kl"].items()}
        for b in ks:
            if b + 1 in ks and b + 1 <= 6 and ks[b] > 1e-9:
                RS.setdefault((m, typ(a)), []).append(ks[b + 1] / ks[b])


def r_loo(m, t):
    v = [x for (mm, tt), xs in RS.items() if tt == t and mm != m for x in xs] or \
        [x for (mm, tt), xs in RS.items() if mm != m for x in xs]
    return float(np.median(v))


for m, (meta, L, T, DN, KL_ALL) in DATA.items():
    FILE0 = meta["file_bytes_est"]
    print("\n=== %s  KL_ALL(live) %.6f  FILE %.1f МБ" % (m, KL_ALL, FILE0 / 1e6))
    _, FILE = greedy(T, KL_ALL, FILE0, 1e9)
    # шаги вниз
    steps = []
    for a, e in DN.items():
        ks = {int(b): k for b, k in e["kl"].items()}
        for b in sorted(ks):
            if b + 1 in ks and b + 1 <= e["base_bits"]:
                dk = ks[b] - ks[b + 1]; dy = nb(e["params"], b + 1) - nb(e["params"], b)
                steps.append(dict(e=(dk / KL_ALL) / (dy / FILE), a=a, fr=b + 1, to=b, dk=dk, dy=dy, t=typ(a)))
    # A
    for thr in (3, 5):
        s3 = [s for s in steps if s["to"] == 3 and s["e"] < thr]; s4 = [s for s in steps if s["to"] >= 4 and s["e"] < thr]
        by = {}
        for s in s4:
            by[s["t"]] = by.get(s["t"], 0) + s["dy"]
        print("A. e_down<%d: на 3 бита %3d шагов %6.1f МБ (%.2f%% файла); реализуемые (->4) %3d шагов %6.1f МБ %s"
              % (thr, len(s3), sum(s["dy"] for s in s3) / 1e6, 100 * sum(s["dy"] for s in s3) / FILE,
                 len(s4), sum(s["dy"] for s in s4) / 1e6, {k: round(v / 1e6, 1) for k, v in by.items()}))
    # B
    print("B. симметричное правило (вниз -- только реализуемое):")
    print("     tau   вверх: ключей  -KL%%   +МБ | вниз: шагов  +KL%%   -МБ | ИТОГ: KL к live  МБ RAM к live (%% файла)")
    for tau in (3, 4, 5, 6, 7, 10, 15, 20, 30):
        up, _ = greedy(T, KL_ALL, FILE0, tau)
        du, yu = evalT(up, T, L)
        dn = [s for s in steps if s["to"] >= 4 and s["e"] < tau and s["a"] not in up]
        kd = sum(s["dk"] for s in dn); yd = sum(s["dy"] for s in dn)
        ovr = dict(up); ovr.update({s["a"]: s["to"] for s in dn})
        json.dump(ovr, open(f"{D}/sym_{m}_tau{tau}.json", "w"), indent=1)
        print("    %4g        %3d  %6.1f %6.1f |       %3d  %5.1f %6.1f |        %+6.1f%%   %+7.1f (%+.2f%%)"
              % (tau, len(up), 100 * du / KL_ALL, yu / 1e6, len(dn), 100 * kd / KL_ALL, yd / 1e6,
                 100 * (kd - du) / KL_ALL, (yu - yd) / 1e6, 100 * (yu - yd) / FILE))
    # C
    dT, yT = evalT(greedy(T, KL_ALL, FILE0, 5)[0], T, L)
    V1 = {}; V2 = {}
    for a, e in L.items():
        b0 = e["base_bits"]; p = e["params"]; ks = {int(b): k for b, k in e["kl"].items()}; r = r_loo(m, typ(a))
        V1[a] = [(b, ks[b0] * r ** (b - b0), nb(p, b)) for b in range(b0, 7)] + [(16, 0.0, nb(p, 16))]
        if b0 + 1 in ks and b0 + 1 <= 6:
            V2[a] = [(b, ks[b0] if b == b0 else ks[b0 + 1] * r ** (b - b0 - 1), nb(p, b)) for b in range(b0, 7)] + [(16, 0.0, nb(p, 16))]
        else:
            V2[a] = V1[a]
    for name, V in (("V1 база+r", V1), ("V2 база,+1 бит+r", V2)):
        o, _ = greedy(V, KL_ALL, FILE0, 5); d, y = evalT(o, T, L)
        print("C. %-18s снятие %5.1f%% за +%6.1f МБ  (истина: %5.1f%% за +%6.1f МБ); КПД %.2f против %.2f %%/МБ"
              % (name, 100 * d / KL_ALL, y / 1e6, 100 * dT / KL_ALL, yT / 1e6, 100 * d / KL_ALL / (y / 1e6), 100 * dT / KL_ALL / (yT / 1e6)))
print("\nГОТОВО")
