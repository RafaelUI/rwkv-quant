"""04.10: markdown-таблицы статьи из оценок file_eval_0310 (fe0310/*.json, fe_ext/*.json): сводная по шести масштабам
(файлы quantize(), текст + открытый код) и сравнение с GGUF / MLX. Интервалы -- бутстрэп по окнам (20000, сид 20261004).
    python gen_tables_0410.py <каталог srv_0310>"""
import glob, json, math, os, sys
import numpy as np
D = sys.argv[1]; rng = np.random.default_rng(20261004); B = 20000
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def sl(j, k):
    i = [n for n, kk in enumerate(j["kind"]) if kk == k]
    return np.array(j["kl"])[i], np.array(j["ce"])[i], np.array(j["ce_ref"])[i]
def cell(j, k, ci=False):
    kl, ce, cr = sl(j, k); d = 100 * (math.exp(ce.mean() - cr.mean()) - 1)
    if not ci: return "%.5f" % kl.mean(), "%+.2f" % d
    S = rng.integers(0, len(kl), (B, len(kl))); db = 100 * (np.exp(ce[S].mean(1) - cr[S].mean(1)) - 1)
    return "%.5f" % kl.mean(), "%+.2f [%+.2f; %+.2f]" % (d, *np.percentile(db, [2.5, 97.5]))
def ratio(a, b, k):
    ka = sl(a, k)[0]; kb = sl(b, k)[0]; S = rng.integers(0, len(ka), (B, len(ka)))
    return "%.2f [%.2f; %.2f]" % (kb.mean() / ka.mean(), *np.percentile(kb[S].mean(1) / ka[S].mean(1), [2.5, 97.5]))
NAMES = (("0p1b", "0.1B"), ("0p4b", "0.4B"), ("1p5b", "1.5B"), ("2p9b", "2.9B"), ("7p2b", "7.2B"), ("13b", "13.3B"))
ARMS = (("REDUCTION", "rtn_red"), ("REDUCTION + GPTQ (умолчание)", "red_gptq"), ("COMPRESSION", "rtn_comp"), ("+ автоподбор", "rtn_wprop"),
        ("+ GPTQ", "comp_gptq_s0"), ("+ оба (умолчание)", "comp_both_s0"))
print("| модель | плечо | МБ | KL текст | KL код | Δppl текст, % | Δppl код, % |\n|---|---|---|---|---|---|---|")
for lab, nm in NAMES:
    for title, a in ARMS:
        j = load(os.path.join(D, "fe0310", "%s_%s.json" % (lab, a)))
        if j is None: continue
        kt, dt = cell(j, "text", True); kc, dc = cell(j, "code_open", True)
        b = "**" if "умолчание" in title else ""
        print("| %s | %s%s%s | %.1f | %s | %s | %s | %s |" % (nm, b, title, b, j["bytes"] / 1e6, kt, kc, dt, dc))
print()
EXT = (("gguf_Q4_K_M", "GGUF Q4_K_M (imatrix)"), ("mlx_int4", "MLX int4"), ("mlx_int6", "MLX int6"), ("gguf_Q6_K", "GGUF Q6_K (imatrix)"))
OURS = (("comp_both_s0", "наш COMPRESSION (умолчание)"), ("sym6_rtn", "наш sym6, без GPTQ"), ("sym6_gptq", "наш sym6 + GPTQ"),
        ("sym6h8_rtn", "наш sym6h8, без GPTQ"), ("sym6h8_gptq", "наш sym6h8 + GPTQ"), ("red_gptq", "наш REDUCTION (умолчание)"))
for lab, nm in (("1p5b", "1.5B"), ("2p9b", "2.9B")):
    rows = []
    for a, t in EXT:
        j = load(os.path.join(D, "fe_ext", "ext_%s_%s.json" % (lab, a)))
        if j: rows.append((j["bytes"], t, j, a))
    for a, t in OURS:
        j = load(os.path.join(D, "fe0310", "%s_%s.json" % (lab, a)))
        if j: rows.append((j["bytes"], "**" + t + "**", j, a))
    print("**%s**\n\n| формат | МБ | KL текст | KL код | Δppl текст, %% | Δppl код, %% |\n|---|---|---|---|---|---|" % nm)
    R = {}
    for by, t, j, a in sorted(rows, key=lambda r: r[0]):
        kt, dt = cell(j, "text", True); kc, dc = cell(j, "code_open", True); R[a] = j
        print("| %s | %.1f | %s | %s | %s | %s |" % (t, by / 1e6, kt, kc, dt, dc))
    print("\nОтношения KL (чужой / наш, парный бутстрэп по окнам; > 1 -- наш лучше):\n")
    for e, o in (("gguf_Q4_K_M", "comp_both_s0"), ("mlx_int4", "comp_both_s0"), ("mlx_int6", "sym6_gptq"), ("mlx_int6", "sym6_rtn"), ("gguf_Q6_K", "sym6h8_gptq"),
                 ("gguf_Q6_K", "sym6h8_rtn"), ("mlx_int6", "comp_both_s0"), ("gguf_Q6_K", "red_gptq"), ("mlx_int6", "red_gptq")):
        if e in R and o in R:
            print("- %s / %s (размер нашего %+.1f%%): текст %s, код %s" % (dict(EXT)[e], dict(OURS)[o], 100 * (R[o]["bytes"] / R[e]["bytes"] - 1), ratio(R[o], R[e], "text"), ratio(R[o], R[e], "code_open")))
    print()
