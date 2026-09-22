"""ПРАВИЛО ОТ БАЗЫ ПРЕСЕТА (вариант (а), решение владельца 22.09): emb стартует с 6,
o_proj слоя 0 -- с bf16, KL_ALL -- у пресета (совместная проверка). Лестницы те же
(одна матрица квантована, остальное fp32 -- от базы не зависят).
    python rule_a_2209.py <ladder.json> <KL_пресета> <TAU>  -> последняя строка OVR {...}"""
import json, sys
path, KL_ALL, TAU = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
lad = json.load(open(path)); meta, L = lad["_meta"], lad["ladder"]
nb = lambda p, b: 2 * p if b >= 16 else p * (b + 0.5) / 8
START = {"emb.weight": 6, "blocks.0.att.output.weight": 16}
FILE = meta["file_bytes_est"]
lv, cur = {}, {}
for arm, e in L.items():
    bs = sorted(int(b) for b in e["kl"])
    lv[arm] = [(b, e["kl"][str(b)], nb(e["params"], b)) for b in bs]
    s = START.get(arm, bs[0]); cur[arm] = [b for b, _, _ in lv[arm]].index(s)
    FILE += lv[arm][cur[arm]][2] - lv[arm][0][2]
taken = []; cb = cr = ck = 0.0
while True:
    best = None
    for a, lev in lv.items():
        b0, k0, y0 = lev[cur[a]]
        for j in range(cur[a] + 1, len(lev)):
            b1, k1, y1 = lev[j]; db, dkl = y1 - y0, k0 - k1
            e = (dkl / KL_ALL) / (db / FILE) if db > 0 else 0
            if best is None or e > best[0]: best = (e, a, j, b0, b1, dkl, db)
    if best is None or best[0] < TAU: break
    e, a, j, b0, b1, dkl, db = best; cur[a] = j
    ram = 0 if a == "emb.weight" else db
    taken.append((a, b0, b1, e, dkl, db)); cb += db; cr += ram; ck += dkl
    print("   %-30s %2d->%-2d e %6.1f  dKL %5.1f%%  +%.2f МБ" % (a, b0, b1, e, 100*dkl/KL_ALL, db/1e6))
print("ИТОГ %s TAU %g: шагов %d, +%.1f МБ RAM (%.2f%%), KL %.4f -> %.4f (%.0f%%, предсказание)" % (
    path.split("_")[1], TAU, len(taken), cr/1e6, 100*cr/FILE, KL_ALL, KL_ALL-ck, 100*(KL_ALL-ck)/KL_ALL))
print("OVR " + json.dumps({a: b1 for a, _, b1, *_ in taken}))
