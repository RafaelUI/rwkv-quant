"""ПОЛНАЯ ИСТИНА ПРАВИЛА (22.09 ночь). Вход: ws_<m>_ladder_full.json (прежняя лестница +
добор всех плеч со строгой границей e>=5), ws_<m>_proxy.json, ws_<m>_joint.json (live),
ws_<m>_down.json (если есть), rule_files_a/ovr_<m>_tau5.json (если есть).
  1  выбор на полной лестнице при tau=5 против (а); кривая KL/байты по tau
  2  предфильтр признаком: e_proxy = (score/KL_ALL)/(байты +1 бит/FILE) >= tau/kappa
     + принудительные (emb, head, слой 0, последний) -- доля ключей выбора, выживших
     в кандидатах, и число кандидатов (цена measure)
  3  лестница вниз: e_down = (dKL/KL_ALL)/(снятые байты/FILE) по возрастанию; баланс
     "tau=5 вверх + вниз до нуля байт к live" (предсказание по одиночным, joint -- отдельно)
    python autopick_full_2209.py [каталог=~/rwkvq]"""
import json, os, sys
import numpy as np
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/rwkvq")
START = {"emb.weight": 6, "blocks.0.att.output.weight": 16}
nb = lambda p, b: 2 * p if b >= 16 else p * (b + 0.5) / 8   # noqa: E731


def greedy(lad, KL_ALL, FILE0, TAU):
    lv, cur, FILE = {}, {}, FILE0
    for a, lev in lad.items():
        bs = [b for b, _, _ in lev]
        cur[a] = bs.index(START.get(a, bs[0])); lv[a] = lev
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
    ovr = {a: lv[a][cur[a]][0] for a in lv if cur[a] != st[a]}
    drop = sum(lv[a][st[a]][1] - lv[a][cur[a]][1] for a in ovr)
    add = sum(lv[a][cur[a]][2] - lv[a][st[a]][2] for a in ovr)
    ram = sum(lv[a][cur[a]][2] - lv[a][st[a]][2] for a in ovr if a != "emb.weight")
    return ovr, drop, add, ram, FILE


OUT = {}
for m in ["0p1b", "0p4b", "1p5b", "2p9b", "7p2b", "13b"]:
    f = f"{D}/ws_{m}_ladder_full.json"
    if not os.path.exists(f):
        continue
    lad = json.load(open(f)); meta, L = lad["_meta"], lad["ladder"]
    prox = json.load(open(f"{D}/ws_{m}_proxy.json"))["rows"]
    KL_ALL = json.load(open(f"{D}/ws_{m}_joint.json"))["joint"]["live"]["kl"]
    FILE0, Lyr = meta["file_bytes_est"], meta["L"]
    incomplete = [a for a, e in L.items() if not ("16" in e["kl"] and str(e["base_bits"]) in e["kl"])]
    T = {a: sorted((int(b), k, nb(e["params"], int(b))) for b, k in e["kl"].items()) for a, e in L.items() if a not in incomplete}
    print("\n=== %s  KL_ALL(live) %.6f  FILE %.1f МБ  лестница %d (незаконченных %d)" % (m, KL_ALL, FILE0 / 1e6, len(T), len(incomplete)))
    o = OUT.setdefault(m, {})
    # 1
    ovr5, d5, a5, r5, FILE = greedy(T, KL_ALL, FILE0, 5)
    fa = f"{D}/rule_files_a/ovr_{m}_tau5.json"
    if os.path.exists(fa):
        A = json.load(open(fa))
        TA = {a: T[a] for a in A}
        dA = sum(dict((b, k) for b, k, _ in T[a])[START.get(a, L[a]["base_bits"])] - dict((b, k) for b, k, _ in T[a])[A[a]] for a in A)
        rA = sum(nb(L[a]["params"], A[a]) - nb(L[a]["params"], START.get(a, L[a]["base_bits"])) for a in A if a != "emb.weight")
        print("1. (а) записанный: %2d ключей, снятие %5.1f%%, +%6.1f МБ RAM" % (len(A), 100 * dA / KL_ALL, rA / 1e6))
        o["a"] = dict(n=len(A), drop=dA / KL_ALL, ram_mb=rA / 1e6)
    print("   полный tau=5:   %2d ключей, снятие %5.1f%%, +%6.1f МБ RAM (%.2f%% файла); новых ключей %d"
          % (len(ovr5), 100 * d5 / KL_ALL, r5 / 1e6, 100 * r5 / FILE, len(set(ovr5) - set(json.load(open(fa)) if os.path.exists(fa) else {}))))
    o["full5"] = dict(n=len(ovr5), drop=d5 / KL_ALL, ram_mb=r5 / 1e6, ovr=ovr5)
    json.dump(ovr5, open(f"{D}/ovr_full_{m}_tau5.json", "w"), indent=1)
    print("   кривая: tau  ключей  снятие%%  +МБ RAM")
    curve = []
    for tau in (40, 20, 10, 7, 5, 3, 2):
        ov, d, a, r, _ = greedy(T, KL_ALL, FILE0, tau)
        curve.append((tau, len(ov), d / KL_ALL, r / 1e6)); json.dump(ov, open(f"{D}/ovr_full_{m}_tau{tau}.json", "w"), indent=1)
        print("        %4g  %5d  %6.1f  %7.1f" % (tau, len(ov), 100 * d / KL_ALL, r / 1e6))
    o["curve"] = curve
    # 2
    forced = lambda a: a in ("emb.weight", "head.weight") or a.startswith("blocks.0.") or a.startswith("blocks.%d." % (Lyr - 1))   # noqa: E731
    print("2. предфильтр признаком (цель -- %d ключей полного выбора tau=5)" % len(ovr5))
    pe = {}
    for a, r in prox.items():
        if a.endswith(".lora") or "score" not in r:
            continue
        if a in L:
            p, b = L[a]["params"], L[a]["base_bits"]
        else:
            continue_ = True
            p = None
        if p is None:
            continue
        pe[a] = (r["score"] / KL_ALL) / ((nb(p, b + 1) - nb(p, b)) / FILE)
    # для плеч вне лестницы параметры -- по форме (как в autopick_offline)
    C, V = meta["C"], 65536
    for a, r in prox.items():
        if a in pe or a.endswith(".lora") or "score" not in r:
            continue
        t = a.split(".", 2)[-1]
        p = V * C if a in ("emb.weight", "head.weight") else (4 * C * C if t.startswith("ffn.") else C * C)
        b = 5 if "ffn.key" in a or a in ("emb.weight", "head.weight") else 4
        pe[a] = (r["score"] / KL_ALL) / ((nb(p, b + 1) - nb(p, b)) / FILE)
    rows = []
    for kappa in (1, 1.5, 2, 3, 5, 8):
        cand = {a for a, e in pe.items() if e >= 5 / kappa or forced(a)}
        hit = [a for a in ovr5 if a in cand]; miss = [a for a in ovr5 if a not in cand]
        lost = sum(dict((b, k) for b, k, _ in T[a])[START.get(a, L[a]["base_bits"])] - dict((b, k) for b, k, _ in T[a])[ovr5[a]] for a in miss)
        rows.append((kappa, len(cand), len(hit), lost / KL_ALL, miss[:3]))
        print("   kappa %-4g кандидатов %3d из %3d; выжило %2d/%2d; потеряно снятия %4.1f%% KL; пропуски %s"
              % (kappa, len(cand), len(pe), len(hit), len(ovr5), 100 * lost / KL_ALL, miss[:3]))
    o["prefilter"] = rows
    # 3
    fd = f"{D}/ws_{m}_down.json"
    if os.path.exists(fd) and "down" in json.load(open(fd)):
        DN = json.load(open(fd))["down"]
        steps = []
        for a, e in DN.items():
            if a in ovr5:
                continue            # правило подняло -- вниз не трогаем
            ks = {int(b): k for b, k in e["kl"].items()}
            b0 = e["base_bits"]
            for b in sorted(ks, reverse=True):
                if b < b0 and b + 1 in ks:
                    dk = ks[b] - ks[b + 1]; dy = nb(e["params"], b + 1) - nb(e["params"], b)
                    steps.append(((dk / KL_ALL) / (dy / FILE), a, b + 1, b, dk, dy))
        steps.sort()
        print("3. вниз: %d плеч, %d шагов; самые дешёвые:" % (len(DN), len(steps)))
        for s in steps[:8]:
            print("     %-32s %d->%d  e %6.2f  +KL %5.2f%%  -%.2f МБ" % (s[1], s[2], s[3], s[0], 100 * s[4] / KL_ALL, s[5] / 1e6))
        # жадно: шаг b->b-1 допустим, только если b уже достигнут (сначала b0->b0-1)
        cur = {}; cum_y = cum_k = 0.0; target = r5; got = None; table = []
        pending = list(steps)
        while pending:
            for i, s in enumerate(pending):
                e, a, bfrom, bto, dk, dy = s
                if cur.get(a, DN[a]["base_bits"]) == bfrom:
                    break
            else:
                break
            pending.pop(i); cur[a] = bto; cum_y += dy; cum_k += dk
            table.append((e, cum_y, cum_k))
            if got is None and cum_y >= target:
                got = (len(table), e, cum_y, cum_k)
        for thr in (0.5, 1, 2, 3, 5):
            pts = [t for t in table if t[0] < thr]
            if pts:
                print("     e_down < %-4g снимается %7.1f МБ (%5.2f%% файла) за +%5.1f%% KL" % (thr, pts[-1][1] / 1e6, 100 * pts[-1][1] / FILE, 100 * pts[-1][2] / KL_ALL))
        if got:
            n, e, y, k = got
            print("   БАЛАНС: вернуть +%.1f МБ правила = %d шагов вниз (последний e %.2f), +%.1f%% KL; итог к live: KL %+.1f%%, байты ~0"
                  % (r5 / 1e6, n, e, 100 * k / KL_ALL, 100 * (k - d5) / KL_ALL))
            o["balance"] = dict(steps=n, e_last=e, kl_up=k / KL_ALL, net=(k - d5) / KL_ALL)
        o["down_table"] = [(round(t[0], 3), t[1] / 1e6, t[2] / KL_ALL) for t in table]
json.dump(OUT, open(f"{D}/autopick_full_2209.json", "w"), indent=1, ensure_ascii=False)
print("\nГОТОВО")
