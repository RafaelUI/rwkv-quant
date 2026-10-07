"""07.10: сквозная проверка calibrate() после правки доводки -- настоящий прогон с умолчаниями, затем quantize(config=).
    python e2e_calibrate_0710.py <ckpt> <eval.pt> <device> <out.json>"""
import json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant import calibrate, quantize
CK, EV, DEV, OUT = sys.argv[1:5]
t0 = time.time()
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    cfg = calibrate(CK, EV, device=DEV, verbose=True)
    rep = cfg.calibration_report
    f = OUT.replace(".json", ".rwkvq")
    quantize(CK, f, config=cfg, tokenizer=None, verbose=False, allow_per_row=True)
r = dict(delta=rep["combined_delta_pct"], met=rep["budget_met"], final=rep["final"], exhausted=rep["exhausted"],
         steps=rep["refine_steps"], mb=os.path.getsize(f) / 1e6, sec=time.time() - t0, warnings=[str(w.message)[:200] for w in W])
json.dump(r, open(OUT, "w"), ensure_ascii=False, indent=1)
print("ИТОГ: композит %+.2f%%, бюджет %s, файл %.2f МБ, %.0f с, предупреждений %d" % (r["delta"], "выполнен" if r["met"] else "НЕ выполнен", r["mb"], r["sec"], len(r["warnings"])))
print(r["final"]); print(r["warnings"])
