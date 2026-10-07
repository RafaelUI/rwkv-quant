"""Гейт доводки композита в calibrate() (07.10, п. 2 сессии).

Было: при ppl_threshold_pct = 5 calibrate() отдавал композит +8.9...+10.4% (0.1B-2.9B): в доводке сосед той же
цены считался тупиком, и proj / cmix / emb / head выбывали, не попробовав следующую битность; о невыполненном
бюджете сообщал print под verbose с неверным текстом, а quantize(config=) молчал.
Стало: schema_space.refine (чистая функция, редакция V4 пробы) + warnings.warn + предупреждение в quantize().

Данные: tests/data/calib_refine_0p1b.json -- ЗАПИСАННАЯ таблица прогонов ppl настоящей 0.1B (сервер, cuda,
eval_text_heldout, проба tests/_sess/p_calib_budget_0710.py): состояние {группа: кандидат} -> Δ% композита, для
состояний, которые посещают редакции V0 / V2 / V3 / V4. Шаг за пределы таблицы -- провал свойства (не подмена числа).

Свойства:
  P1 на записанной таблице refine повторяет записанную траекторию V4 шаг в шаг (группа, кандидат, Δ, принят/откат);
  P2 бюджет 5% достигнут, итог совпадает с записанным; за пределы таблицы не выходил; записанный итог прежнего
     правила (V0) -- выше бюджета (то, что чинили);
  P3 размер квантуемых групп равен записанному V4 и меньше, чем у редакции без обновления прокси (V2);
  P4 принятые шаги строго улучшают композит;
  P5 (синтетика -- проверяет управление, а не числа): a) сосед той же цены не исчерпывает группу;
     b) если ничто не помогает -- остановка, бюджет не достигнут, состояние не тронуто; c) прокси обновляется после
     шага: вторая группа получает свой подъём, первая не уезжает вверх;
  P6 _budget_warning: None в бюджете; вне бюджета -- текст с Δ, порогом и исчерпанными группами; quantize(config=)
     с отчётом выше порога предупреждает ДО отказов и работы, с отчётом в бюджете и без отчёта -- молчит;
  P7 (настоящая модель 0.1B, ~3 мин; SKIP_MODEL=1 пропускает): calibrate(verbose=False) с недостижимым бюджетом
     выдаёт UserWarning и budget_met = False.
Мутации (--mutate): равная цена = тупик (P2); прокси не обновляется (P3); шаг принимается без улучшения (P5);
предупреждение calibrate выключено (P6); предупреждение quantize выключено (P6).
    python tests/test_calibrate_refine.py [--mutate]"""
import inspect, json, os, sys, tempfile, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rwkv_quant import api
from rwkv_quant.calibration import schema_space as S
from rwkv_quant.calibration.group_config import QuantConfig

HERE = os.path.dirname(os.path.abspath(__file__))
FX = json.load(open(os.path.join(HERE, "data", "calib_refine_0p1b.json"), encoding="utf-8"))
T = FX["table"]
CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
EV = os.path.expanduser(os.environ.get("EV", "~/Develop/WKV-kvant/eval_text_heldout.pt"))
TH = FX["th"]


class C:
    def __init__(self, r, e): self.r, self.eff_bits = r, e
    def __repr__(self): return self.r


CANDS = {g: [C(c["repr"], c["eff_bits"]) for c in cs] for g, cs in T["cands"].items()}
VAR = {v["var"]: v for v in FX["variants"]}
skey = lambda st: json.dumps(sorted((g, (-1 if i is None else i)) for g, i in st.items()))
name = lambda g, j: repr(CANDS[g][j]) if j is not None else "bf16"
size = lambda cur: sum(T["numel"][g] * (CANDS[g][cur[g]].eff_bits if cur[g] is not None else 16.0) for g in CANDS) / 8e6


def on_table():
    miss = []
    def measure(state):
        full = {g: None for g in CANDS}; full.update(state)
        k = skey(full)
        if k not in T["runs"]:
            miss.append(k); return float("inf")
        return T["runs"][k]
    r = S.refine(CANDS, dict(T["chosen_i"]), dict(FX["iso"]), FX["delta_before"], measure,
                 lambda g, i: measure({g: i}), TH)
    return r, miss


def synth(cands, cur, iso, oracle, th, iso_of=None):
    return S.refine(cands, cur, iso, oracle(cur), oracle, iso_of or (lambda g, i: 0.0), th)


def props(d):
    R = []
    def check(n, ok, info=""): R.append((n, bool(ok), str(info)[:300]))

    r, miss = on_table()
    got = [(s["group"], name(s["group"], s["to"]), round(s["delta_pct"], 3), "принят" if s["accepted"] else "откат") for s in r["steps"]]
    want = [tuple(t) for t in VAR["V4"]["trace"]]
    check("P1 траектория == записанной V4 (%d шагов)" % len(want), got == want and len(want) >= 5,
          [x for x in zip(got, want) if x[0] != x[1]][:1] or (len(got), len(want)))
    check("P2 бюджет достигнут, итог == записанному", r["met"] and r["delta"] <= TH and abs(r["delta"] - VAR["V4"]["delta"]) < 1e-9, r["delta"])
    check("P2 за пределы таблицы не выходил", not miss, miss[:1])
    check("P2 прежнее правило (V0) на этой модели -- выше бюджета", VAR["V0"]["delta"] > TH, VAR["V0"]["delta"])
    check("P3 размер == записанному V4 и меньше V2", abs(size(r["cur"]) - VAR["V4"]["mb_groups"]) < 0.05 and size(r["cur"]) < VAR["V2"]["mb_groups"] - 1,
          (size(r["cur"]), VAR["V4"]["mb_groups"], VAR["V2"]["mb_groups"]))
    acc = [FX["delta_before"]] + [s["delta_pct"] for s in r["steps"] if s["accepted"]]
    check("P4 принятые шаги строго улучшают", all(b < a for a, b in zip(acc, acc[1:])) and len(acc) > 1, acc)

    # P5a
    ca = {"A": [C("a0", 4.5), C("a1", 4.5), C("a2", 5.5), C("a3", 6.5)]}
    ra = synth(ca, {"A": 0}, {"A": 10.0}, lambda st: {0: 10.0, 1: 10.5, 2: 4.0, 3: 1.0, None: 0.0}[st["A"]], 5.0)
    check("P5a сосед той же цены не исчерпывает группу", ra["met"] and ra["cur"]["A"] == 2 and "A" not in ra["exhausted"], ra)
    # P5b
    rb = synth(ca, {"A": 0}, {"A": 10.0}, lambda st: 10.0, 5.0)
    check("P5b ничто не помогает -> стоп, не достигнут, состояние прежнее",
          not rb["met"] and rb["cur"] == {"A": 0} and rb["n_steps"] <= S.REFINE_MAX_STEPS and not any(s["accepted"] for s in rb["steps"]), rb)
    # P5c
    cc = {"A": [C("a0", 4.5), C("a1", 5.5), C("a2", 6.5)], "B": [C("b0", 4.5), C("b1", 5.5)]}
    ia = {0: 4.0, 1: 0.1, 2: 0.05, None: 0.0}; ib = {0: 2.0, 1: 0.2, None: 0.0}
    rc = synth(cc, {"A": 0, "B": 0}, {"A": 4.0, "B": 2.0}, lambda st: ia[st["A"]] + ib[st["B"]], 0.5,
               iso_of=lambda g, i: (ia if g == "A" else ib)[i])
    check("P5c прокси обновляется: A=1, B=1", rc["met"] and rc["cur"] == {"A": 1, "B": 1}, rc["cur"])

    # P6
    check("P6 в бюджете -- None", api._budget_warning(4.9, 5.0, {"proj": "x"}, [], 3) is None)
    m = api._budget_warning(9.13, 5.0, {"proj": "sb6@4", "small": "bf16"}, ["cmix", "head"], 7) or ""
    check("P6 вне бюджета -- текст с Δ, порогом, группами", all(x in m for x in ("+9.13%", "5.0%", "НЕ достигнут", "cmix, head", "proj=sb6@4")) and "small" not in m, m[:200])
    def q_warns(rep):
        c = QuantConfig(proj=4)                       # построчные 4 бита -> быстрый отказ ПОСЛЕ точки предупреждения
        if rep is not None: c.calibration_report = rep
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter("always")
            try: api.quantize(CK, os.path.join(d, "x.rwkvq"), tokenizer=None, config=c, verbose=False)
            except ValueError: pass
        return [str(w.message) for w in W if "calibrate()" in str(w.message)]
    w_over = q_warns({"combined_delta_pct": 9.1, "threshold_pct": 5.0})
    check("P6 quantize(config=) выше бюджета -- предупреждает", len(w_over) == 1 and "+9.10%" in w_over[0], w_over)
    check("P6 quantize(config=) в бюджете / без отчёта -- молчит", not q_warns({"combined_delta_pct": 4.1, "threshold_pct": 5.0}) and not q_warns(None))

    # P7
    if os.environ.get("SKIP_MODEL") != "1":
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter("always")
            cfg = api.calibrate(CK, EV, device=os.environ.get("DEV", "mps"), ppl_threshold_pct=-1.0, groups=["w_lora"], verbose=False)
        ws = [str(w.message) for w in W if "НЕ достигнут" in str(w.message)]
        rep = cfg.calibration_report
        check("P7 calibrate(verbose=False), недостижимый бюджет -> UserWarning и budget_met=False",
              len(ws) == 1 and rep["budget_met"] is False and rep["combined_delta_pct"] > -1.0, (ws[:1], rep.get("budget_met")))
    return R


def run(label):
    with tempfile.TemporaryDirectory() as d:
        R = props(d)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        os.environ["SKIP_MODEL"] = "1"      # мутации ловятся без модели; P7 -- в контрольных прогонах
        SRC = inspect.getsource(S.refine)
        def mut_src(a, b):
            assert SRC.count(a) == 1, a
            ns = {}; exec(SRC.replace(a, b), S.__dict__, ns); S.refine = ns["refine"]
        muts = [("равная цена = тупик", lambda: mut_src('            gain = float("inf") if free else iso[g] / cost\n',
                                                         '            if free:\n                exhausted.add(g)\n                continue\n            gain = iso[g] / cost\n'), "P2"),
                ("прокси не обновляется", lambda: mut_src("            iso[g] = measure_iso(g, j)\n", "            pass\n"), "P3"),
                ("шаг принимается без улучшения", lambda: mut_src("        ok = d_new < delta - 1e-9\n", "        ok = True\n"), "P5"),
                ("предупреждение calibrate выключено", lambda: setattr(api, "_budget_warning", lambda *a, **k: None), "P6"),
                ("предупреждение quantize выключено", lambda: setattr(api, "_warn_calibration_over_budget", lambda *a, **k: None), "P6")]
        for nm, apply, expect in muts:
            saved = (S.refine, api._budget_warning, api._warn_calibration_over_budget)
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % nm)
            finally:
                S.refine, api._budget_warning, api._warn_calibration_over_budget = saved
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        os.environ.pop("SKIP_MODEL")
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
