"""07.10: почему доводка композита в calibrate() останавливается выше бюджета, и что дают варианты правки.
Код библиотеки НЕ меняется: изолированная стадия скопирована из api.calibrate как есть, доводка -- в четырёх редакциях
на ОДНОМ изолированном отборе (прогоны ppl кешируются по конфигу, так что одинаковые шаги у редакций -- одно число):
  V0  как в api.calibrate сейчас (обязана воспроизвести записанный композит): сосед той же цены (add_bits <= 0) --
      группа считается исчерпанной;
  V1  «прыжок»: следующий кандидат -- первый СТРОГО дороже текущего (соседи той же цены пропускаются);
  V2  «бесплатный шаг»: сосед той же цены пробуется первым (байты те же); не помог -- указатель идёт дальше, группа
      не исчерпана; платный шаг -- как раньше;
  V3  V1 + прокси выигрыша обновляется: после подъёма iso_delta группы -- изолированная Δ НОВОГО кандидата (доп. прогон).
  V4  V3 + бесплатный шаг V2 (добавлено по вопросу владельца 07.10): сосед той же цены пробуется первым, прокси обновляется.
V1-V4: предел шагов 60 вместо 20. PCB_VARS=V3,V4 -- выбрать редакции.
    python p_calib_budget_0710.py <ckpt> <eval.pt> <device> <out.json> [порог=5]"""
import json, os, sys, time
if "/Develop/rwkv-quant/tests" in os.path.abspath(__file__): sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant import api as A
from rwkv_quant.calibration import QuantConfig, GROUPS
from rwkv_quant.calibration.outlier_scan import GROUP_KEY_PATTERNS, LORA_BIAS_SUFFIXES
from rwkv_quant.calibration import schema_space as S
from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK, EV, DEV, OUT = sys.argv[1:5]
TH = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
T0 = time.time()
model = RWKV7Ref(CK, device=DEV, dtype=torch.bfloat16)
data = A._load_corpus(EV, DEV)
sd = torch.load(CK, map_location="cpu", mmap=True)
base = perplexity(model, data, QuantConfig())
RUNS = {}
def delta_of(kw):
    key = json.dumps([sorted(kw["bits"].items()), sorted(kw["group_scale"].items()), sorted(kw["group_scale_mode"].items())])
    if key not in RUNS:
        RUNS[key] = 100 * (perplexity(model, data, A._mk_config(kw, None)) - base) / base
    return RUNS[key]
def empty(): return {"bits": {g: 16 for g in GROUPS}, "group_scale": {}, "group_scale_mode": {}}
def iso_delta(g, c):
    t = empty(); c.apply_to(t, g); return delta_of(t)
def numel(g):
    n = 0
    for k, t in sd.items():
        if getattr(t, "dim", None) is None or not any(k.endswith(p) or p in k for p in GROUP_KEY_PATTERNS[g]): continue
        if k.endswith(LORA_BIAS_SUFFIXES) or t.dim() < 2: continue
        n += t.numel()
    return n
# ---- изолированная стадия: копия api.calibrate ----
kw0, CANDS, CI, ISO, NUM = empty(), {}, {}, {}, {}
for g in GROUPS:
    inf, all_2d = S.group_shapes(sd, g, GROUP_KEY_PATTERNS[g]); NUM[g] = numel(g)
    if inf == 0: CANDS[g], CI[g], ISO[g] = [], None, 0.0; continue
    CANDS[g] = S.candidates_for(inf, have_act_stats=False, all_2d=all_2d); CI[g] = None; ISO[g] = 0.0
    for ci, c in enumerate(CANDS[g]):
        d = iso_delta(g, c)
        if d <= TH: CI[g], ISO[g] = ci, d; c.apply_to(kw0, g); break
N_ISO = len(RUNS); T_ISO = time.time() - T0
D0 = delta_of(kw0)
def size_mb(cur):
    return sum(NUM[g] * (CANDS[g][cur[g]].eff_bits if cur[g] is not None else 16.0) for g in GROUPS) / 8e6
print("ckpt %s  baseline %.4f  порог %.1f%%  изолированно %d прогонов %.0f с  композит до доводки %+.2f%%" % (os.path.basename(CK), base, TH, N_ISO, T_ISO, D0))
print("отбор:", {g: (repr(CANDS[g][CI[g]]) if CI[g] is not None else "bf16", round(ISO[g], 2)) for g in GROUPS}, flush=True)
def refine(var):
    t0, n0 = time.time(), len(RUNS)
    kw = {k: dict(v) for k, v in kw0.items()}
    cur = dict(CI); ptr = dict(CI); iso = dict(ISO); exhausted, trace = set(), []
    d_all, guard, lim = D0, 0, (20 if var == "V0" else 60)
    while d_all > TH and guard < lim:
        guard += 1; best = None
        for g in GROUPS:
            if g in exhausted or kw["bits"][g] >= 16 or cur[g] is None: continue
            cs = CANDS[g]; i = cur[g]
            if var in ("V1", "V3"):  # V4 идёт по указателю, как V2
                j = next((x for x in range(i + 1, len(cs)) if cs[x].eff_bits > cs[i].eff_bits + 1e-9), None)
            else:
                j = ptr[g] + 1 if ptr[g] + 1 < len(cs) else None
            nxt = cs[j] if j is not None else None
            add = (nxt.eff_bits if nxt else 16.0) - cs[i].eff_bits
            if add <= 0:
                if var == "V0": exhausted.add(g); continue
                gain = float("inf")                      # V2: бесплатный шаг пробуется первым
            else:
                gain = iso[g] / add
            if best is None or gain > best[1]: best = (g, gain, nxt, j, add)
        if best is None: break
        g, gain, nxt, j, add = best
        prev = {k: dict(v) for k, v in kw.items()}
        if nxt is None: kw["bits"][g] = 16; kw["group_scale"].pop(g, None); kw["group_scale_mode"].pop(g, None)
        else: nxt.apply_to(kw, g)
        d_new = delta_of(kw); ok = d_new < d_all - 1e-9
        trace.append((g, repr(nxt) if nxt else "bf16", round(d_new, 3), "принят" if ok else "откат"))
        if not ok:
            kw = prev
            if var in ("V2", "V4") and add <= 0: ptr[g] = j      # бесплатный не помог -- идём дальше, группа жива
            else: exhausted.add(g)
            continue
        cur[g] = ptr[g] = j; d_all = d_new
        if var in ("V3", "V4") and nxt is not None: iso[g] = iso_delta(g, nxt)
    return dict(var=var, delta=d_all, reached=d_all <= TH, steps=guard, runs=len(RUNS) - n0, sec=time.time() - t0,
                mb_groups=size_mb(cur), final={g: (repr(CANDS[g][cur[g]]) if cur[g] is not None else "bf16") for g in GROUPS},
                exhausted=sorted(exhausted), trace=trace)
RES = dict(ckpt=CK, baseline=base, th=TH, n_iso=N_ISO, sec_iso=T_ISO, delta_before=D0, mb_before=size_mb(CI),
           chosen={g: (repr(CANDS[g][CI[g]]) if CI[g] is not None else "bf16") for g in GROUPS}, iso=ISO, variants=[])
for var in os.environ.get("PCB_VARS", "V0,V1,V2,V3").split(","):
    r = refine(var); RES["variants"].append(r); json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("== %s: композит %+.2f%% (%s), шагов %d, новых прогонов %d, %.0f с, квантуемые группы %.1f МБ (до доводки %.1f)" % (
        var, r["delta"], "в бюджете" if r["reached"] else "ВЫШЕ бюджета", r["steps"], r["runs"], r["sec"], r["mb_groups"], RES["mb_before"]))
    for t in r["trace"]: print("     %-7s -> %-36s %+8.3f%%  %s" % t)
    print("   итог:", r["final"], "| исчерпаны:", r["exhausted"], flush=True)
print("ВСЕГО %.0f с" % (time.time() - T0))
