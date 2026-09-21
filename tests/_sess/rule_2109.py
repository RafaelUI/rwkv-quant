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
Декод ограничен полосой: мс/токен ~ dB / 66.8 ГБ/с (замер 17.09, 2.9B).
    python rule_2109.py <ladder.json> <KL_всей_модели> [TAU]
"""
import json, sys
lad = json.load(open(sys.argv[1]))
meta, L = lad["_meta"], lad["ladder"]
KL_ALL = float(sys.argv[2]); TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
FILE = meta["file_bytes_est"]; BW = 66.8e9
nb = lambda p, b: 2 * p if b >= 16 else p * (b + 0.5) / 8   # noqa: E731

steps = {}
for arm, e in L.items():
    bs = sorted(int(b) for b in e["kl"])
    seq = []
    for a, b in zip(bs, bs[1:]):
        dkl = e["kl"][str(a)] - e["kl"][str(b)]
        db = nb(e["params"], b) - nb(e["params"], a)
        seq.append(dict(arm=arm, frm=a, to=b, dkl=dkl, db=db,
                        e=(dkl / KL_ALL) / (db / FILE) if db > 0 else 0.0,
                        ram=0 if arm == "emb.weight" else db))
    steps[arm] = seq

pos = {a: 0 for a in steps}
taken, cum_b, cum_ram, cum_kl = [], 0.0, 0.0, 0.0
print("модель %s  KL всей модели %.6f  файл ~%.1f МБ  TAU %.1f" % (
    meta["ckpt"].split("/")[-1][:24], KL_ALL, FILE / 1e6, TAU))
print(" #  матрица                          шаг     e     dKL%   +МБ файл  +МБ RAM  +мс/ток  KL после")
while True:
    nxt = [(steps[a][pos[a]], a) for a in steps if pos[a] < len(steps[a])]
    if not nxt:
        break
    s, a = max(nxt, key=lambda t: t[0]["e"])
    if s["e"] < TAU:
        break
    pos[a] += 1; taken.append(s)
    cum_b += s["db"]; cum_ram += s["ram"]; cum_kl += s["dkl"]
    print("%2d  %-32s %2d->%-2d %6.1f  %5.1f%%  %7.2f  %7.2f  %7.3f  %.6f" % (
        len(taken), s["arm"], s["frm"], s["to"], s["e"], 100 * s["dkl"] / KL_ALL,
        s["db"] / 1e6, s["ram"] / 1e6, 1e3 * s["ram"] / BW, KL_ALL - cum_kl))
print("ИТОГ: шагов %d, +%.1f МБ файла (%.2f%%), +%.1f МБ RAM, декод +%.3f мс/ток, "
      "KL %.6f -> %.6f (предсказание, %.0f%%)" % (
          len(taken), cum_b / 1e6, 100 * cum_b / FILE, cum_ram / 1e6, 1e3 * cum_ram / BW,
          KL_ALL, KL_ALL - cum_kl, 100 * (KL_ALL - cum_kl) / KL_ALL))
ovr = {}
for s in taken:
    ovr[s["arm"]] = s["to"]
print("OVR " + json.dumps(ovr))
