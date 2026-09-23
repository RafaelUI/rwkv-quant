"""ОФЛАЙН-ПРОВЕРКА ДЛЯ measure (22.09, ночь). Только JSON сервера, без модели.
Вопросы:
  0  согласованы ли свип (истина на базе, все плечи) и лестница на базовом уровне;
  A  воспроизводит ли жадный выбор на истинной лестнице записанный ovr_*_tau5 (а);
  B  ПОЛНОТА КАНДИДАТОВ: у плеча вне лестницы шаг +k бит не снимет больше его
     одиночного вреда S => e <= (S/KL_ALL)/(байты +1 бит/FILE). Граница СТРОГАЯ в
     определении самого правила (ладдер на базе == свип, проверка 0);
  C  закон уровней: r = KL(b+1)/KL(b) по типам матриц и масштабам;
  D  выбор по ОЦЕНЁННЫМ лестницам (база x r^k, r -- медиана по ДРУГИМ масштабам):
     D1 база = истина свипа (measure = 1 проход на плечо);
     D2 база = признак rel2*G для всех;
     D3 признак + истина для принудительных (emb, head, слой 0, последний слой);
     оценка выбора -- по истинной лестнице, где она есть (иначе помечено).
    python autopick_offline_2209.py [каталог=~/rwkvq]   (env TAU=5)"""
import glob, json, os, sys
import numpy as np
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/rwkvq")
TAU = float(os.environ.get("TAU", "5"))
SCALES = ["0p1b", "0p4b", "1p5b", "2p9b", "7p2b", "13b"]
START = {"emb.weight": 6, "blocks.0.att.output.weight": 16}
V = 65536
nb = lambda p, b: 2 * p if b >= 16 else p * (b + 0.5) / 8   # noqa: E731


def typ(a):
    if a in ("emb.weight", "head.weight"):
        return a.split(".")[0]
    t = a.split(".", 2)[2].rsplit(".", 1)[0]
    return "proj" if t.startswith("att.") else "cmix." + t.split(".")[1]


def params(a, C):
    t = a.split(".", 2)[-1]
    if a in ("emb.weight", "head.weight"):
        return V * C
    return 4 * C * C if t.startswith("ffn.") else C * C


def load(m):
    lad = json.load(open(f"{D}/ws_{m}_ladder.json"))
    prox = json.load(open(f"{D}/ws_{m}_proxy.json"))["rows"]
    J = json.load(open(f"{D}/ws_{m}_joint.json"))["joint"]
    S = {}
    for f in sorted(glob.glob(f"{D}/sweep_{m}_compression_nol0_s*.json")):
        d = json.load(open(f)); S.update({k: v["kl"] for k, v in d.items() if not k.startswith("_")})
    for f in sorted(glob.glob(f"{D}/ws_{m}_sweep*.json")):
        S.update({k: v["kl"] for k, v in json.load(open(f)).get("rows", {}).items()})
    fd = f"{D}/ws_{m}_down.json"          # 13.3B: полного свипа нет -- база из режима down (тот же прибор)
    if os.path.exists(fd):
        for k, e in json.load(open(fd)).get("down", {}).items():
            S.setdefault(k, e["kl"][str(e["base_bits"])])
    S = {k: v for k, v in S.items() if not k.startswith("__") and not k.endswith(".lora") and v is not None}
    return lad["_meta"], lad["ladder"], prox, J, S


def greedy(lad, KL_ALL, FILE0):
    """rule_a_2209.py один в один: lad[arm] = [(b, kl, bytes)] по возрастанию b."""
    lv, cur, FILE = {}, {}, FILE0
    for a, lev in lad.items():
        bs = [b for b, _, _ in lev]
        cur[a] = bs.index(START.get(a, bs[0])); lv[a] = lev
        FILE += lev[cur[a]][2] - lev[0][2]
    start = dict(cur); taken = []
    while True:
        best = None
        for a, lev in lv.items():
            b0, k0, y0 = lev[cur[a]]
            for j in range(cur[a] + 1, len(lev)):
                b1, k1, y1 = lev[j]; db, dkl = y1 - y0, k0 - k1
                e = (dkl / KL_ALL) / (db / FILE) if db > 0 else 0
                if best is None or e > best[0]:
                    best = (e, a, j, b0, b1, dkl, db)
        if best is None or best[0] < TAU:
            break
        e, a, j, b0, b1, dkl, db = best; cur[a] = j; taken.append((a, b0, b1, e))
    ovr = {a: lv[a][cur[a]][0] for a in lv if cur[a] != start[a]}
    return ovr, taken, FILE


def true_ladder(L):
    return {a: sorted((int(b), k, nb(e["params"], int(b))) for b, k in e["kl"].items()) for a, e in L.items()}


def synth(base_kl, b0, p, r):
    lev = [(b, base_kl * r ** (b - b0), nb(p, b)) for b in range(b0, 7)]
    return lev + [(16, 0.0, nb(p, 16))]


# ---------------- загрузка и закон уровней (C) по всем масштабам
DATA, RS = {}, {}
for m in SCALES:
    if not os.path.exists(f"{D}/ws_{m}_ladder.json"):
        continue
    DATA[m] = load(m)
    for a, e in DATA[m][1].items():
        ks = {int(b): k for b, k in e["kl"].items()}
        for b in sorted(ks):
            if b + 1 in ks and b + 1 <= 6 and ks[b] > 1e-9:
                RS.setdefault((m, typ(a)), []).append(ks[b + 1] / ks[b])
TYPES = ["proj", "cmix.key", "cmix.value", "head", "emb"]
print("C. r = KL(b+1)/KL(b), медиана [мин; макс] (n)")
print("   %-5s" % "" + "".join("%-24s" % t for t in TYPES))
for m in DATA:
    row = ""
    for t in TYPES:
        v = RS.get((m, t), [])
        row += "%-24s" % ("%.3f [%.2f;%.2f] (%d)" % (np.median(v), min(v), max(v), len(v)) if v else "-")
    print("   %-5s%s" % (m, row))


def r_loo(m, t):
    v = [x for (mm, tt), xs in RS.items() if tt == t and mm != m for x in xs]
    if not v:
        v = [x for (mm, tt), xs in RS.items() if mm != m for x in xs]
    return float(np.median(v))


REPORT = {}
for m, (meta, L, prox, J, S) in DATA.items():
    C, Lyr, FILE0 = meta["C"], meta["L"], meta["file_bytes_est"]
    KL_ALL = J["live"]["kl"]
    T = true_ladder(L)
    base = {}
    for a, e in L.items():
        base.setdefault(typ(a), set()).add(e["base_bits"])
    assert all(len(v) == 1 for v in base.values()), base
    base = {t: v.pop() for t, v in base.items()}
    base.setdefault("proj", 4); base.setdefault("cmix.value", 4); base.setdefault("cmix.key", 5)
    bb = lambda a: base[typ(a)]                                           # noqa: E731
    for a, e in L.items():
        assert e["params"] == params(a, C), (a, e["params"], params(a, C))
    print("\n=== %s  L%d C%d  KL_ALL(live) %.6f  FILE %.1f МБ  лестница %d, признак %d, свип %d"
          % (m, Lyr, C, KL_ALL, FILE0 / 1e6, len(L), len(prox), len(S)))
    rep = REPORT.setdefault(m, {})

    # 0. свип против лестницы на базе
    both = [a for a in L if a in S]
    if both:
        d = [abs(L[a]["kl"][str(L[a]["base_bits"])] - S[a]) / max(S[a], 1e-12) for a in both]
        print("0. лестница(база) vs свип: %d общих, max отн. расхождение %.2e" % (len(both), max(d)))
        rep["check0_maxrel"] = max(d)

    # A. воспроизведение (а)
    ovrT, takenT, FILE = greedy(T, KL_ALL, FILE0)
    f = f"{D}/rule_files_a/ovr_{m}_tau{TAU:g}.json"
    if os.path.exists(f):
        rec = json.load(open(f))
        print("A. выбор по истинной лестнице == записанный ovr: %s (%d ключей)" % (rec == ovrT, len(ovrT)))
        rep["A_equal"] = rec == ovrT
    else:
        print("A. записанного ovr нет; выбор по истинной лестнице: %d ключей" % len(ovrT))
    emin = min(t[3] for t in takenT) if takenT else None
    print("   последний взятый шаг e = %s" % ("%.2f" % emin if emin else "-"))

    # B. полнота кандидатов
    out = [a for a in S if a not in L]
    if out:
        rows = []
        for a in out:
            p, b = params(a, C), bb(a)
            db = nb(p, b + 1) - nb(p, b)
            eb = (S[a] / KL_ALL) / (db / FILE)
            r = r_loo(m, typ(a))
            rows.append((eb, eb * (1 - r), a, S[a]))
        rows.sort(reverse=True)
        nb_ = sum(1 for x in rows if x[0] >= TAU); ne = sum(1 for x in rows if x[1] >= TAU)
        print("B. вне лестницы %d плеч: граница e >= %g у %d, оценка e(+1 бит) >= %g у %d" % (len(out), TAU, nb_, TAU, ne))
        for eb, ee, a, s in rows[:6]:
            print("     %-32s S %.6f  граница %6.2f  оценка %6.2f" % (a, s, eb, ee))
        json.dump([a for eb, _, a, _ in rows if eb >= TAU], open(f"{D}/cand_{m}_bound_tau{TAU:g}.json", "w"), indent=0)
        rep["B"] = dict(n_out=len(out), n_bound=nb_, n_est=ne, top=[(a, round(eb, 2), round(ee, 2)) for eb, ee, a, _ in rows[:6]])
    else:
        print("B. свипа нет -- полнота не проверяется")

    # D. выбор по оценённым лестницам
    arms = [a for a in prox if not a.endswith(".lora") and a in {**L, **S} or (not a.endswith(".lora") and a in prox)]
    arms = [a for a in dict.fromkeys(list(prox) + list(L)) if not a.endswith(".lora")]
    forced = [a for a in arms if a in ("emb.weight", "head.weight") or a.startswith("blocks.0.") or a.startswith("blocks.%d." % (Lyr - 1))]

    def truth_base(a):
        if a in L:
            return L[a]["kl"][str(L[a]["base_bits"])]
        return S.get(a)

    variants = {}
    if len(S) >= 0.9 * len([a for a in arms if a not in ("head.weight",)]):
        variants["D1 истина базы"] = {a: truth_base(a) for a in arms}
    variants["D2 признак"] = {a: prox[a]["score"] for a in arms if a in prox}
    variants["D3 признак+принуд."] = {a: (truth_base(a) if a in forced else prox[a]["score"]) for a in arms if a in prox}

    def evaluate(ovr):
        drop, unv, byt = 0.0, 0, 0.0
        for a, bf in ovr.items():
            p, b0 = params(a, C), bb(a)
            s = START.get(a, b0)
            if a in T:
                kk = {b: k for b, k, _ in T[a]}
                drop += kk[s] - kk[bf]
            else:
                unv += 1
                x = truth_base(a) if truth_base(a) is not None else prox[a]["score"]
                r = r_loo(m, typ(a))
                ks = lambda b: 0.0 if b >= 16 else x * r ** (b - b0)   # noqa: E731
                drop += ks(s) - ks(bf)
            byt += nb(p, bf) - nb(p, s)
        return drop, byt, unv

    dT, yT, _ = evaluate(ovrT)
    print("D. эталон (истинная лестница): %2d ключей, снятие KL %5.1f%% за +%.1f МБ" % (len(ovrT), 100 * dT / KL_ALL, yT / 1e6))
    rep["D0"] = dict(n=len(ovrT), drop=dT / KL_ALL, mb=yT / 1e6)
    for name, X in variants.items():
        lad = {a: synth(X[a], bb(a), params(a, C), r_loo(m, typ(a))) for a in X if X[a] is not None}
        ovr, taken, _ = greedy(lad, KL_ALL, FILE0)
        d, y, unv = evaluate(ovr)
        inter = set(ovr.items()) & set(ovrT.items()); keys = set(ovr) & set(ovrT)
        print("   %-20s %2d ключей, снятие %5.1f%% за +%6.1f МБ; совпало ключ+бит %d, ключ %d из %d; без истины %d; лишние %s; пропущены %s"
              % (name, len(ovr), 100 * d / KL_ALL, y / 1e6, len(inter), len(keys), len(ovrT), unv,
                 sorted(set(ovr) - set(ovrT))[:4], sorted(set(ovrT) - set(ovr))[:4]))
        rep[name] = dict(n=len(ovr), drop=d / KL_ALL, mb=y / 1e6, same_kb=len(inter), same_k=len(keys), unverified=unv)

json.dump(REPORT, open(f"{D}/autopick_offline_2209.json", "w"), indent=1, ensure_ascii=False)
print("\nГОТОВО -> %s/autopick_offline_2209.json" % D)
