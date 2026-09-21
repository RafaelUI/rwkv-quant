"""ПРАВИЛО "СЛАБОЕ МЕСТО -> БИТ" (21.09) поверх лестниц weakspot_2109 ladder.

Шаг = +1 бит одной матрице (или последний шаг 6 -> bf16). Для шага:
  dKL   = KL(одна матрица в b) - KL(одна матрица в b')     (истинный KL, 8 окон)
  dB    = байты(b') - байты(b)
  e     = (dKL / KL_всей_модели) / (dB / байты_файла)       -- "доля качества
          на долю размера"; e = 1 значит: байты тратятся с той же отдачей, что
          средний байт файла.
Жадно: всегда берётся следующий доступный шаг (b -> b+1 у той же матрицы) с
наибольшим e, пока e >= TAU. Выход: порядок шагов с накопленными байтами,
предсказанным KL, ценой в RAM и трафиком декода.
ЦЕНА: RAM и чтение на токен растут ровно на dB -- КРОМЕ emb (в COMPRESSION он
в памяти плотный fp16 и читается одной строкой: 0 к RAM и 0 к декоду).
Скорость декода НЕ выводится из байтов (деквант 5-6 бит бывает быстрее 4, у\nгрупп своя полоса) -- только замер ABBA на Mac.
    python rule_2109.py <ladder.json> <KL_всей_модели> [TAU]
"""
import json, sys
lad = json.load(open(sys.argv[1]))
meta, L = lad["_meta"], lad["ladder"]
KL_ALL = float(sys.argv[2]); TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
FILE = meta["file_bytes_est"]; BW = 66.8e9
nb = lambda p, b: 2 * p if b >= 16 else p * (b + 0.5) / 8   # noqa: E731

# ПРЫЖКИ, а не только соседние шаги: у 2.9B o_proj слоя 0 4->5 даёт 0.3%,
# а 5->6 -- 22% (e 6.5 против 496). Жадный "по ближайшему шагу" при TAU 10
# туда не доходил. Для матрицы на уровне b кандидаты -- ВСЕ уровни b' > b,
# e(b->b') по накопленным dKL и dB; берётся лучший прыжок (выпуклая
# оболочка лестницы).
lv = {}
for arm, e in L.items():
    bs = sorted(int(b) for b in e["kl"])
    lv[arm] = [(b, e["kl"][str(b)], nb(e["params"], b)) for b in bs]

cur = {a: 0 for a in lv}
taken, cum_b, cum_ram, cum_kl = [], 0.0, 0.0, 0.0
print("модель %s  KL всей модели %.6f  файл ~%.1f МБ  TAU %.1f" % (
    meta["ckpt"].split("/")[-1][:24], KL_ALL, FILE / 1e6, TAU))
print(" #  матрица                          шаг     e     dKL%   +МБ файл  +МБ RAM  KL после")
while True:
    best = None
    for a, levels in lv.items():
        b0, k0, y0 = levels[cur[a]]
        for j in range(cur[a] + 1, len(levels)):
            b1, k1, y1 = levels[j]
            db, dkl = y1 - y0, k0 - k1
            e = (dkl / KL_ALL) / (db / FILE) if db > 0 else 0.0
            if best is None or e > best[0]:
                best = (e, a, j, b0, b1, dkl, db)
    if best is None or best[0] < TAU:
        break
    e, a, j, b0, b1, dkl, db = best
    cur[a] = j
    ram = 0 if a == "emb.weight" else db
    taken.append(dict(arm=a, frm=b0, to=b1, e=e, dkl=dkl, db=db, ram=ram))
    cum_b += db; cum_ram += ram; cum_kl += dkl
    print("%2d  %-32s %2d->%-2d %6.1f  %5.1f%%  %7.2f  %7.2f  %.6f" % (
        len(taken), a, b0, b1, e, 100 * dkl / KL_ALL, db / 1e6, ram / 1e6, KL_ALL - cum_kl))
print("ИТОГ: шагов %d, +%.1f МБ файла (%.2f%%), +%.1f МБ RAM (+%.2f%%), "
      "KL %.6f -> %.6f (предсказание, %.0f%%)" % (
          len(taken), cum_b / 1e6, 100 * cum_b / FILE, cum_ram / 1e6, 100 * cum_ram / FILE,
          KL_ALL, KL_ALL - cum_kl, 100 * (KL_ALL - cum_kl) / KL_ALL))
ovr = {}
for s in taken:
    ovr[s["arm"]] = s["to"]
print("OVR " + json.dumps(ovr))
