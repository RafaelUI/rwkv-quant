"""28.09: эталон ДО/ПОСЛЕ правки нулевого суперблока в _gw_one (qm = 0/0 -> NaN).
    python gw_zero_sb_freeze_2809.py freeze|check
freeze -- хеши (md5 сырых байт) частей _gw_one и упаковки writer (_make_qt_gw_sb6, _make_qt_gw_asym)
на реальных тензорах 0.1B g1d и blocks.4.ffn.key 1.5B g1j (почти мёртвые строки, |max| до 1e-14),
bits 4/5/6, search да/нет, ex2 есть/нет (сид zlib.crc32 -- закон 28) -> /tmp/gwfix/ref.json.
check -- то же после правки: реальные тензоры ОБЯЗАНЫ совпасть побитно; синтетика с нулевым
суперблоком -- до правки NaN, после -- конечна, нулевой суперблок деквантуется в 0, прочие строки побитно."""
import hashlib, json, sys, zlib
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
from rwkv_quant.calibration import groupwise as gw
from rwkv_quant.formats import writer as W
MODE = sys.argv[1]
REF = "/tmp/gwfix/ref.json"


def h(t):
    t = t.detach().cpu().contiguous()
    return hashlib.md5(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes()).hexdigest()


def hq(qt):
    out = {}
    for k, v in sorted(vars(qt).items()):
        if isinstance(v, torch.Tensor):
            out[k] = h(v)
        elif isinstance(v, (int, float, str, tuple, list)) or v is None:
            out[k] = str(v)
    return out


def ex_for(name, n):
    g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
    return torch.rand(n, generator=g) + 0.1


sd0 = torch.load("/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth", mmap=True, map_location="cpu")
sd1 = torch.load("/Users/s/Develop/WKV-kvant/rwkv7-g1j-1.5b-20260831-ctx16384.pth", mmap=True, map_location="cpu")
T = {k: sd0[k] for k in ("blocks.0.att.receptance.weight", "blocks.3.ffn.key.weight", "blocks.5.ffn.value.weight", "blocks.2.att.output.weight")}
T["g1j.blocks.4.ffn.key.weight"] = sd1["blocks.4.ffn.key.weight"][6912:8192]   # полоса с почти мёртвыми строками
res = {}
for name, w in T.items():
    for bits in (4, 5, 6):
        for search in (True, False):
            for use_ex in (True, False):
                ex = ex_for(name, w.shape[1]) if use_ex else None
                tag = "%s|%d|%s|%s" % (name, bits, search, use_ex)
                p = gw._gw_one(w.float().clone(), bits, 32, 8, -6 if search else 6, ex, True)
                res["gw1|" + tag] = {k: h(v) for k, v in sorted(p.items())}
                res["sb6|" + tag] = hq(W._make_qt_gw_sb6(name, "cmix", bits, w, 32, ex, search=search))
    res["asym|" + name] = hq(W._make_qt_gw_asym(name, "cmix", 4, w, 32))
# синтетика
torch.manual_seed(0)
ws = torch.randn(4, 2048) * 0.02
ws[1, 256:512] = 0.0; ws[2] = 0.0; ws[3, :256] = 1e-9
syn = {}
for bits in (4, 5, 6):
    o = gw._gw_one(ws.clone(), bits, 32, 8, -6, ex_for("syn", 2048), False)
    syn[bits] = dict(nonfin=(~torch.isfinite(o)).sum(1).tolist(), row0=h(o[0]), row3=h(o[3]),
                     zero_sb_is_zero=bool((o[1, 256:512] == 0).all()) if torch.isfinite(o[1, 256:512]).all() else False,
                     zero_row_is_zero=bool((o[2] == 0).all()) if torch.isfinite(o[2]).all() else False,
                     row1_rest=h(torch.cat([o[1, :256], o[1, 512:]])))
if MODE == "freeze":
    json.dump(dict(real=res, syn=syn), open(REF, "w"), indent=0)
    print("заморожено: %d записей; синтетика до правки:" % len(res), {b: s["nonfin"] for b, s in syn.items()})
else:
    ref = json.load(open(REF))
    bad = [k for k in res if res[k] != ref["real"][k]]
    print("реальные тензоры: %d записей, расхождений %d %s" % (len(res), len(bad), bad[:5]))
    for b in (4, 5, 6):
        s, r = syn[b], ref["syn"][str(b)]
        print("синтетика %d бит: нечисел %s (было %s); нулевой суперблок -> 0: %s; нулевая строка -> 0: %s; строки 0 и 3 побитно: %s; строка 1 вне нулевого суперблока побитно: %s" % (
            b, s["nonfin"], r["nonfin"], s["zero_sb_is_zero"], s["zero_row_is_zero"], s["row0"] == r["row0"] and s["row3"] == r["row3"], s["row1_rest"] == r["row1_rest"]))
    ok = not bad and all(sum(syn[b]["nonfin"]) == 0 and syn[b]["zero_sb_is_zero"] and syn[b]["zero_row_is_zero"] for b in (4, 5, 6))
    print("ИТОГ:", "ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ")
