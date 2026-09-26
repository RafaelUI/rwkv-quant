"""Гейт разнесения measure по картам (24.09): 0.1B g1d, одна карта против двух ("cuda:1,cuda:2").
Утверждения: KL пресета и KL каждого плеча по каждому окну совпадают (отн. 1e-9; ожидается
побитно -- те же кернели на тех же GPU); выбор select_budget(+0.5%) совпадает.
Только сервер с >= 3 картами:  python tests/test_measure_multidev.py <ckpt> <tok>"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap
CK, TOK = sys.argv[1], sys.argv[2]
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
m1 = ap.measure(CK, cfg, TOK, device="cuda:1", cache=False)
m2 = ap.measure(CK, cfg, TOK, device="cuda:1,cuda:2", cache=False)
dev = abs(m2["kl_all"] / m1["kl_all"] - 1)
bit = m1["kl_all_w"] == m2["kl_all_w"]
for k, a in m1["arms"].items():
    for b, v in a["klw"].items():
        w = m2["arms"][k]["klw"][b]
        bit = bit and v == w
        dev = max(dev, max(abs(x / y - 1) for x, y in zip(w, v)))
same = ap.select_budget(m1, 0.005)[0] == ap.select_budget(m2, 0.005)[0]
print("макс. отн. расхождение %.2e, побитно %s, выбор совпал %s, время %.0f / %.0f с" % (dev, bit, same, m1["seconds"], m2["seconds"]))
ok = dev < 1e-9 and same
print("ИТОГ:", "ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ")
sys.exit(0 if ok else 1)
