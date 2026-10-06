"""07.10 (п. 3): фактическая таблица «режим x биты x блок x форма -> что делает писатель СЕЙЧАС».
Тензоры -- настоящие из 0.1B (по одному на каждую различную форму в группе; у широких OUT обрезан до 64 строк:
условия поддержки зависят от IN и числа осей, а не от OUT -- обрезка названа, закон 17).
Исход: ok:<раскладка> либо <Исключение>: <начало текста>. Пути: real_gw=True (файл), real_gw=False (fake),
fake_quant.q (то, чем меряет calibrate / perplexity).
    python p3_mode_grid_0710.py <ckpt> <out.json>"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant.calibration import QuantConfig, GROUPS
from rwkv_quant.calibration.groupwise import GW_MODES
from rwkv_quant.calibration import fake_quant as FQ
from rwkv_quant.formats import writer as W
CK, OUT = sys.argv[1:3]
sd = torch.load(CK, map_location="cpu", mmap=True)
reps = collections.OrderedDict()
for k, w in sd.items():
    g = W._match_group(k)
    if not W._is_quantized(k, g, w.dim()): continue
    reps.setdefault((g, tuple(w.shape)), k)
print("представители:"); [print("  %-8s %-22s %s" % (g, sh, k)) for (g, sh), k in reps.items()]
MODES = [None] + list(GW_MODES) + ["sym_typo", "bogus"]
BITS = [1, 2, 3, 4, 5, 6, 7, 8]
GS = [16, 32, 64, 33]
def outcome(fn):
    try:
        r = fn()
        if isinstance(r, torch.Tensor): return "ok:dense"
        return "ok:" + (getattr(r, "gw_mode", None) or ("dense" if r.dense is not None else "rtn"))
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt): raise
        return "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:70])
RES = []
for (g, sh), k in reps.items():
    w = sd[k].float()
    if w.dim() == 2 and w.shape[0] > 64: w = w[:64].contiguous()
    for mode in MODES:
        for gs in GS:
            for b in BITS:
                def cfg():
                    c = QuantConfig(group_scale={g: gs}, group_scale_mode=({g: mode} if mode else {}), strict=False, **{g: b})
                    return c
                r = dict(group=g, shape=list(sh), key=k, mode=mode or "(нет: asym)", gs=gs, bits=b)
                r["real"] = outcome(lambda: W.quantize_tensor(k, w, cfg(), real_gw=True))
                r["fake"] = outcome(lambda: W.quantize_tensor(k, w, cfg(), real_gw=False))
                r["fq"] = outcome(lambda: FQ.q(w, g, cfg(), key=k))
                RES.append(r)
    json.dump(RES, open(OUT, "w"), ensure_ascii=False)
    print("готово:", g, sh, flush=True)
# ---- свёртка: класс формы -> режим -> gs -> строка исходов по битам 1..8 ----
def cls(r):
    sh = r["shape"]
    if len(sh) != 2: return "не 2-D %s" % (tuple(sh),)
    return "2-D IN=%d (IN%%256=%d)" % (sh[1], sh[1] % 256)
def short(o):
    return o if o.startswith("ok") else o.split(":")[0].replace("NotImplementedError", "NotImpl").replace("AssertionError", "Assert")
for path in ("real", "fake", "fq"):
    print("\n================ путь %s ================" % path)
    tab = collections.OrderedDict()
    for r in RES: tab.setdefault((cls(r), r["mode"], r["gs"]), {}).setdefault(r["bits"], set()).add(short(r[path]))
    for (c, m, gs), d in tab.items():
        print("%-26s %-16s gs=%-3d | %s" % (c, m, gs, "  ".join("%d:%s" % (b, "/".join(sorted(d[b]))) for b in BITS)))
print("\nтексты исключений (real):")
for t, n in collections.Counter(r["real"] for r in RES if not r["real"].startswith("ok")).most_common(40): print("  %5d  %s" % (n, t))
