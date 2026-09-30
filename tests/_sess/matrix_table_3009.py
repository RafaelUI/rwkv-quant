"""30.09, СЕРВЕР: сводная таблица деградации из JSON cc4/cc5 + budget_eval_2809 + cc2/cc3 (все -- один прибор: те же 60
окон eval_text_heldout + eval_code_heldout, KL к bf16 тем же RWKV7Ref). Размер -- реальные файлы matrix_files_3009
(GPTQ-файл того же размера, что RTN-файл того же конфига). Сверка прибора: плечо rtn каждого GPTQ-JSON обязано
совпасть с тем же плечом matrix/budget_eval (печатается расхождение). Пустая клетка -- замера нет.
    python matrix_table_3009.py > matrix_table_3009.md"""
import json, math, os, re
R = os.path.expanduser("~/rwkvq/")
MODELS = [("0p1b", "0.1B", "0.1b"), ("0p4b", "0.4B", "0.4b"), ("1p5b", "1.5B", "1.5b"), ("2p9b", "2.9B", "2.9b"),
          ("7p2b", "7.2B", "7.2b"), ("13b", "13.3B", "13.3b")]
CK = {"0p1b": "rwkv7-g1d-0.1b-20260129-ctx8192.pth", "0p4b": "rwkv7-g1d-0.4b-20260210-ctx8192.pth"}
for l, _, s in MODELS[2:]:
    CK[l] = "rwkv7-g1j-%s-20260831-ctx16384.pth" % s
load = lambda f: json.load(open(R + f)) if os.path.exists(R + f) else None
sizes = {}
if os.path.exists(R + "files_3009.log"):
    for m in re.finditer(r"РАЗМЕР (\S+) (\S+) (\d+) байт", open(R + "files_3009.log").read()):
        sizes[(m.group(1), m.group(2))] = int(m.group(3))


def agg(d, arm):
    k = d["kind"]; kk = ["text" if x in ("text", "eval") else "code" for x in k]
    mean = lambda xs, t: sum(x for x, y in zip(xs, kk) if y == t) / kk.count(t)
    ppl = lambda n, t: math.exp(mean(d["ce"][n], t))
    return dict(klt=mean(d["kl"][arm], "text"), klc=mean(d["kl"][arm], "code"),
                dpt=100 * (ppl(arm, "text") / ppl("ref", "text") - 1), dpc=100 * (ppl(arm, "code") / ppl("ref", "code") - 1))


rows, checks = [], []
for lab, name, s in MODELS:
    mx = load("matrix_%s.json" % lab)
    be = load("budget_eval_%s_2809.json" % s)
    src = {}
    if mx:
        for a in mx["kl"]:
            src[a] = agg(mx, a)
    if be and "comp" not in src:
        src["comp"] = agg(be, "live"); src["wprop"] = agg(be, "wprop_+0.005")
    for key, f, base in (("comp+gptq", "gptq_%s_c2w600.json" % lab, "comp"), ("wprop+gptq", "gptq_%s_c2w600_wprop.json" % lab, "wprop"),
                         ("red+gptq", "gptq_red_%s.json" % lab, "red")):
        g = load(f)
        if g and "gptq" in g["kl"]:          # идущий прогон пишет JSON с одним плечом rtn
            src[key] = agg(g, "gptq")
            r = agg(g, "rtn")
            if base in src:
                checks.append("%s %s: rtn %.6f / %.6f против %s %.6f / %.6f" % (name, key, r["klt"], r["klc"], base, src[base]["klt"], src[base]["klc"]))
            else:
                src[base] = r
    bf = os.path.getsize(R + "ckpt/" + CK[lab])
    for arm, preset, sz in (("red", "REDUCTION", "red"), ("red+gptq", "REDUCTION + GPTQ (прототип)", "red"),
                            ("comp", "COMPRESSION (RTN)", "comp"), ("wprop", "COMPRESSION + autopick", "wprop"),
                            ("comp+gptq", "COMPRESSION + GPTQ", "comp"), ("wprop+gptq", "COMPRESSION + autopick + GPTQ (умолчание)", "wprop")):
        v = src.get(arm)
        b = sizes.get((lab, sz))
        rows.append((name, preset, "%.0f" % (b / 1e6) if b else "", "%.2f" % (100 * b / bf) if b else "",
                     "%+.2f" % v["dpt"] if v else "", "%+.2f" % v["dpc"] if v else "",
                     "%.5f" % v["klt"] if v else "", "%.5f" % v["klc"] if v else ""))
    rows.append((name, "bf16 (исходный)", "%.0f" % (bf / 1e6), "100", "0", "0", "0", "0"))
print("| модель | пресет | МБ | % от bf16 | Δppl текст, % | Δppl код, % | KL текст | KL код |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
print("\nСверка прибора (плечо rtn GPTQ-прогона против того же плеча отдельного замера):")
for c in checks:
    print("- " + c)
