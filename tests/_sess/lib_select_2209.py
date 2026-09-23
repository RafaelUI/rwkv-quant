"""Сквозная проверка дешёвого measure (22.09 ночь): ИЗМЕРЕНИЕ собирается из истины
(база -- одиночный KL на битности пресета из ladder_full/свипа; вниз -- из ws_down),
выбор -- библиотечным autopick.select; сравнение с выбором по полной истине (sym_*).
Пишет lib_<m>_tau<t>.json (ovr поверх live).   python lib_select_2209.py [каталог]"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant.calibration import autopick as ap
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/rwkvq")
LIVE = {"emb.weight": 6, "blocks.0.att.output.weight": 16}
for m in ["0p1b", "0p4b", "1p5b", "2p9b", "7p2b", "13b"]:
    if not os.path.exists(f"{D}/ws_{m}_down.json"):
        continue
    fl = f"{D}/ws_{m}_ladder_full.json"
    lad = json.load(open(fl if os.path.exists(fl) else f"{D}/ws_{m}_ladder.json")); meta, L = lad["_meta"], lad["ladder"]
    DN = json.load(open(f"{D}/ws_{m}_down.json"))["down"]
    J = json.load(open(f"{D}/ws_{m}_joint.json"))["joint"]["live"]
    S = {}
    for f in glob.glob(f"{D}/sweep_{m}_compression_nol0_s*.json"):
        S.update({k: v["kl"] for k, v in json.load(open(f)).items() if not k.startswith("_")})
    for f in glob.glob(f"{D}/ws_{m}_sweep*.json"):
        S.update({k: v["kl"] for k, v in json.load(open(f)).get("rows", {}).items()})
    arms = {}
    for k, e in DN.items():                       # все proj/cmix/head, кроме emb и oL0
        b0 = e["base_bits"]; kl = {str(b0): e["kl"][str(b0)]}
        if str(b0 - 1) in e["kl"]:
            kl[str(b0 - 1)] = e["kl"][str(b0 - 1)]
        arms[k] = dict(group=e["group"], params=e["params"], bits=b0, kl=kl)
    e = L["emb.weight"]; arms["emb.weight"] = dict(group="emb", params=e["params"], bits=6, kl={"6": e["kl"]["6"]})
    # байты файла пресета: база nol0 + emb 5->6 + oL0 4->bf16 (как в rule_a)
    oL = L["blocks.0.att.output.weight"]
    FILE = meta["file_bytes_est"] + ap.nbytes(e["params"], 6) - ap.nbytes(e["params"], 5) \
        + ap.nbytes(oL["params"], 16) - ap.nbytes(oL["params"], 4)
    meas = dict(kl_all=J["kl"], file_bytes=FILE, arms=arms)
    # сверка базы down с ladder/свипом (один прибор)
    dif = 0.0 if not S else max(abs(arms[k]["kl"][str(arms[k]["bits"])] - S[k]) / S[k] for k in arms if k in S and S[k] > 0 and k != "emb.weight")   # emb: база пресета 6, свип -- 5
    print("=== %s  арм %d, KL_ALL %.6f, FILE %.1f МБ, база down vs свип max отн. %.1e" % (m, len(arms), J["kl"], FILE / 1e6, dif))
    for t in (5, 6, 7):
        ovr, rep = ap.select(meas, t)
        json.dump(ovr, open(f"{D}/lib_{m}_tau{t}.json", "w"), indent=1)
        fs = f"{D}/sym_{m}_tau{t}.json"
        sym = json.load(open(fs)) if os.path.exists(fs) else {}
        same = sum(1 for k in ovr if sym.get(k) == ovr[k])
        print("   tau %d: вверх %2d вниз %2d, предск. KL %+6.1f%%, байты %+7.1f МБ | полная истина: %2d ключей, совпало ключ+бит %d, "
              "только lib %d, только истина %d"
              % (t, rep["n_up"], rep["n_down"], 100 * rep["kl_pred"], rep["bytes"] / 1e6, len(sym), same,
                 len(set(ovr) - set(sym)), len(set(sym) - set(ovr))))
