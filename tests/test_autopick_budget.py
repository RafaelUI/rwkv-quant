"""Гейт: autopick.select_budget -- выбор при бюджете байт долей файла (24.09).
Данные -- НАСТОЯЩЕЕ измерение (tests/data/measure_g1j_1p5b_rues.json: g1j 1.5B, Mac,
квота en3/ru3/sr2, 145 матриц), не синтетика (закон 17). Утверждения:
  1  байты select(tau) не возрастают с tau, KL-выигрыш не растёт (200 точек log-сетки) --
     на этом стоит бисекция;
  2  для бюджетов -1%, 0, +0.5%, +1%: байты <= бюджета И выбор тугой: при tau на 0.1%
     меньше найденного бюджет превышен (иначе взят не лучший выбор);
  3  согласие с select: бюджет = байты select(5) -> tau <= 5 и KL-выигрыш не хуже;
  4  недостижимый бюджет (-50%) -> feasible False;
  5  РАЗРЕШЕНИЕ: проверка тугости из п. 2 ловит заведомо не лучший выбор (tau x1.5).
    python tests/test_autopick_budget.py"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rwkv_quant.calibration import autopick as ap

M = json.load(open(os.path.join(os.path.dirname(__file__), "data", "measure_g1j_1p5b_rues.json")))
F = M["file_bytes"]
fails = 0
def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print("  %s %s %s" % ("OK  " if ok else "FAIL", name, detail), flush=True)

grid = [math.exp(math.log(1.0) + i * (math.log(100.0) - math.log(1.0)) / 199) for i in range(200)]
B = [ap.select(M, t)[1] for t in grid]
check("1 байты не возрастают с tau", all(B[i + 1]["bytes"] <= B[i]["bytes"] + 1e-6 for i in range(199)))
check("1 KL-выигрыш не растёт с tau", all(B[i + 1]["kl_pred"] >= B[i]["kl_pred"] - 1e-12 for i in range(199)))


def tight(budget, t):
    return ap.select(M, t * 0.999)[1]["bytes"] > budget * F


for b in (-0.01, 0.0, 0.005, 0.01):
    o, r = ap.select_budget(M, b)
    ok = r["feasible"] and r["bytes"] <= b * F and (r["tau"] <= 1.0 + 1e-9 or tight(b, r["tau"]))
    check("2 бюджет %+.3f" % b, ok, "tau %.3f, %+.3f%% файла, KL %+.1f%%" % (r["tau"], 100 * r["bytes_frac"], 100 * r["kl_pred"]))
r5 = ap.select(M, 5.0)[1]
o, r = ap.select_budget(M, r5["bytes"] / F)
check("3 согласие с select(5)", r["tau"] <= 5.0 + 1e-9 and r["kl_pred"] <= r5["kl_pred"] + 1e-12,
      "tau %.3f, KL %+.2f%% против %+.2f%%" % (r["tau"], 100 * r["kl_pred"], 100 * r5["kl_pred"]))
o, r = ap.select_budget(M, -0.5)
check("4 недостижимый бюджет", r["feasible"] is False)
o, r = ap.select_budget(M, 0.005)
bad = r["tau"] * 1.5
check("5 разрешение: не лучший выбор (tau x1.5) пойман", not tight(0.005, bad))
print("ИТОГ:", "ЗЕЛЁНЫЙ" if fails == 0 else "КРАСНЫЙ (%d)" % fails, flush=True)
sys.exit(1 if fails else 0)
