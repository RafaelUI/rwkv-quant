"""06.10: что выберет calibrate(), если убрать из кандидатов построчные 4 бита (schema_space.RTN_BITS = (8,)), и сколько
это стоит в размере файла. Повод: отказ quantize() на построчном RTN ниже 8 бит держит исключение для конфигов calibrate()
(api._is_calibrated); альтернатива -- чтобы calibrate() таких конфигов не выдавал вовсе.
Два прогона calibrate() с умолчаниями на одном чекпоинте: как есть (RTN_BITS = (8, 4)) и без rtn@4.
    python p_calib_rtn4_0610.py <ckpt> <eval.pt> <device> <out.json>"""
import json, os, sys, time
if "/Develop/rwkv-quant/tests" in os.path.abspath(__file__): sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant import calibrate, quantize
from rwkv_quant.calibration import schema_space as S
from rwkv_quant.formats import writer as W
CK, EV, DEV, OUT = sys.argv[1:5]
assert S.RTN_BITS == (8, 4), S.RTN_BITS
RES = {}
for name, rtn in (("как есть", (8, 4)), ("без rtn@4", (8,))):
    S.RTN_BITS = rtn; t0 = time.time()
    cfg = calibrate(CK, EV, device=DEV, verbose=False)
    rep = cfg.calibration_report
    f = "/tmp/p_calib_rtn4_%d.rwkvq" % os.getpid()
    quantize(CK, f, config=cfg, tokenizer=None, verbose=False, allow_per_row=True)
    r = dict(bits=dict(cfg.bits), group_scale=dict(cfg.group_scale), modes=dict(cfg.group_scale_mode),
             chosen={g: v["chosen"] for g, v in rep["groups"].items()},
             first={g: [(t["cand"], round(t["delta_pct"], 2)) for t in v["tried"][:3]] for g, v in rep["groups"].items()},
             rest={k: v for k, v in rep.items() if k != "groups" and isinstance(v, (int, float, str, bool, list))},
             per_row_low=sorted({(g, b) for _, g, b in W.per_row_low_bits(CK, cfg)}), mb=os.path.getsize(f) / 1e6, sec=time.time() - t0)
    RES[name] = r; json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("== %s: файл %.2f МБ, %.0f с; построчно ниже 8 бит: %s" % (name, r["mb"], r["sec"], r["per_row_low"] or "нет"))
    for g, c in r["chosen"].items(): print("   %-7s -> %-34s первые кандидаты: %s" % (g, c, r["first"][g]))
    print("   итог отчёта:", r["rest"], flush=True)
a, b = RES["как есть"], RES["без rtn@4"]
print("РАЗНИЦА: файл %+.2f МБ (%+.2f%%); группы с другим выбором: %s" % (b["mb"] - a["mb"], 100 * (b["mb"] / a["mb"] - 1),
      {g: (a["chosen"][g], b["chosen"][g]) for g in a["chosen"] if a["chosen"][g] != b["chosen"][g]}))
print("файл пробы оставлен:", f)
