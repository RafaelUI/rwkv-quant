"""Гейт autopick.measure (22.09). 0.1B, COMPRESSION, AW-статистика из кеша.
  1  measure отрабатывает (внутри: пол < 1e-9, чистый послойный проход == эталон побитно);
  2  ВОЗОБНОВЛЕНИЕ == ПОЛНЫЙ ПРОХОД: KL выборки плеч (proj L0, att.value L0 -- v_first,
     середина, ffn.key с шагом вниз, последний слой, emb, head) пересчитывается независимым
     M.forward(return_hidden) с подменённой матрицей -- требуется ТОЧНОЕ равенство;
  3  file_bytes в пределах 3% от размера реального файла пресета (если он есть);
  4  select() на этом измерении отрабатывает при tau 5/6/7 и шаги вниз реализуемы (>=4).
    python tests/test_autopick_measure.py"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import autopick as ap, fake_quant
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
ACT = os.path.expanduser(os.environ.get("ACT", "~/.cache/rwkv-quant/act_stats/act_19fa07321aa994e0.pt"))
TOK = os.environ.get("RWKVQ_TOK", "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
REAL = os.path.expanduser(os.environ.get("REAL", "~/Develop/WKV-kvant/compression_0p1b_e6_2109.rwkvq"))
DEV = os.environ.get("RWKVQ_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = ACT
fails = 0
m = ap.measure(CK, cfg, TOK, device=DEV, cache=False, verbose=False)
print("1 measure: %d матриц, KL конфига %.6f, пол %.2e, %.0f с" % (len(m["arms"]), m["kl_all"], m["floor"], m["seconds"]))
# 2: независимый полный проход
from rwkv_quant.calibration import act_stats as A
wins, langs = ap._pick_windows(A._encoder(TOK), A.CORPUS, m["seq_len"])
print("1 окна по языкам: %s" % langs)
fails += langs != [l for l, n in ap.QUOTA for _ in range(n)]
M = RWKV7Ref(CK, device=DEV, dtype=torch.bfloat16, compute_dtype=torch.float32)
data = torch.tensor(wins, dtype=torch.long)[:, :-1].contiguous().to(DEV)
I = ap._Instrument(M, data)
L = M.n_layer
sample = ["blocks.0.att.receptance.weight", "blocks.0.att.value.weight", "blocks.%d.att.output.weight" % (L // 2),
          "blocks.%d.ffn.key.weight" % (L // 2), "blocks.%d.ffn.value.weight" % (L - 1), "emb.weight", "head.weight"]
pts = {p[3]: p for p in ap._points(M)}
with torch.no_grad():
    h_ref = M.forward(data, return_hidden=True)
    for key in sample:
        obj, attr, group, _ = pts[key]
        e = m["arms"][key]
        for b in sorted(int(x) for x in e["kl"]):
            c = copy.copy(cfg); c.bits_overrides = dict({key: b}, **cfg.bits_overrides)
            w = getattr(obj, attr); setattr(obj, attr, fake_quant.q(w.to(M.cdtype), group, c, key))
            if key == "head.weight":
                kl = I.kl(h_ref, h_ref, head=getattr(obj, attr))
            else:
                kl = I.kl(M.forward(data, return_hidden=True), h_ref)
            setattr(obj, attr, w)
            ok = kl == e["kl"][str(b)]
            fails += not ok
            print("2 %-30s %d бит: measure %.9f  полный %.9f  %s" % (key, b, e["kl"][str(b)], kl, "==" if ok else "РАСХОЖДЕНИЕ"))
if os.path.exists(REAL):
    r = os.path.getsize(REAL); d = (m["file_bytes"] - r) / r
    print("3 file_bytes %.2f МБ, реальный файл %.2f МБ (%+.2f%%)" % (m["file_bytes"] / 1e6, r / 1e6, 100 * d))
    fails += abs(d) > 0.03
for t in (5, 6, 7):
    ovr, rep = ap.select(m, t)
    bad = [k for k, b in ovr.items() if b < ap.MIN_REAL]
    print("4 tau %d: вверх %d, вниз %d, байты %+.2f МБ, предск. KL %+.1f%%, нереализуемых %d"
          % (t, rep["n_up"], rep["n_down"], rep["bytes"] / 1e6, 100 * rep["kl_pred"], len(bad)))
    fails += bool(bad)
# 5 (24.09, measure v3): KL по окнам и reweight
n = len(m["langs"])
ok = len(m["kl_all_w"]) == n and ap._mean(m["kl_all_w"]) == m["kl_all"] and all(
    len(v) == n and ap._mean(v) == e["kl"][b] for e in m["arms"].values() for b, v in e["klw"].items())
fails += not ok
print("5 klw: %d окон у каждого плеча, среднее == kl побитно: %s" % (n, ok))
cnt = {l: m["langs"].count(l) for l in set(m["langs"])}
mp = ap.reweight(m, cnt)                      # веса = число окон -> то же измерение
dev = max(abs(mp["arms"][k]["kl"][b] / m["arms"][k]["kl"][b] - 1) for k in m["arms"] for b in m["arms"][k]["kl"])
same = ap.select(mp, 5)[0] == ap.select(m, 5)[0]
fails += not (dev < 1e-12 and same)
print("5 reweight(веса = число окон): макс. отн. отклонение KL %.1e, выбор tau5 совпал: %s" % (dev, same))
m1 = ap.reweight(m, {"en": 1.0})               # только английский -> KL = среднее по окнам en
en = [j for j, l in enumerate(m["langs"]) if l == "en"]
k0 = next(iter(m["arms"]))
exp = ap._mean([m["arms"][k0]["klw"][str(m["arms"][k0]["bits"])][j] for j in en])
ok1 = abs(m1["arms"][k0]["kl"][str(m["arms"][k0]["bits"])] - exp) <= 1e-15 * max(1.0, exp)
fails += not ok1
print("5 reweight({en: 1}) == среднее по окнам en: %s" % ok1)
try:
    ap.reweight(m, {"xx": 1.0}); fails += 1; print("5 вес языка без окон НЕ отвергнут")
except ValueError:
    print("5 вес языка без окон отвергнут")
print("ИТОГ:", "ЗЕЛЁНЫЙ" if not fails else "КРАСНЫЙ (%d)" % fails)
sys.exit(1 if fails else 0)
