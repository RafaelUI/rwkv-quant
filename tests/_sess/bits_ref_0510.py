"""05.10: побитный отпечаток выходов Metal-бэкенда -- до и после правок памяти (должен совпасть).
Для каждого файла: префилл 128 через step, 32 жадных шага по одному (все логиты и состояние), generate в двух режимах,
forward_stateful под mx.compile на 256 токенах. sha256 от байтов fp32.
    python bits_ref_0510.py <out.json> <файл.rwkvq> ..."""
import hashlib, json, os, sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal import generate as G
ev = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt"), weights_only=False)["tokens"]
IDS = ev[12, :256].numpy().astype(np.int32)
def flat(o, acc):
    if isinstance(o, mx.array): acc.append(np.array(o.astype(mx.float32)).tobytes())
    elif isinstance(o, (list, tuple)):
        for x in o: flat(x, acc)
    elif isinstance(o, dict):
        for k in sorted(o): flat(o[k], acc)
def sha(*objs):
    acc = []; flat(list(objs), acc); h = hashlib.sha256()
    for b in acc: h.update(b)
    return h.hexdigest()[:16]
res = {}
for path in sys.argv[2:]:
    m = QuantRWKV7(load_raw(path)); r = {}
    ids = mx.array(IDS[None])
    lg, st = m.step(ids[:, :128], m.init_state(1)); mx.eval(lg, st); r["prefill128"] = sha(lg, st)
    toks, lgs = [], []
    cur = int(np.array(lg[0, -1]).argmax())
    for _ in range(32):
        lg, st = m.step(mx.array([[cur]], dtype=mx.int32), st); mx.eval(lg, st); lgs.append(lg); cur = int(np.array(lg[0, -1]).argmax()); toks.append(cur)
    r["decode32"] = sha(lgs, st); r["tokens"] = toks
    a, _ = G.generate(m, IDS[:64].tolist(), 24); b, _ = G.generate(m, IDS[:64].tolist(), 24, pipeline=False)
    r["generate"] = [a, b]
    lg1, st1 = m.forward_stateful(ids, m.init_state(1), False); mx.eval(lg1, st1); r["fs256"] = sha(lg1, st1)
    step = mx.compile(m.forward_stateful); lg2, st2 = step(ids, m.init_state(1), True); mx.eval(lg2, st2); r["fs256_compiled_last"] = sha(lg2, st2)
    lg3, st3 = m.step(ids[:, :8], m.init_state(1)); mx.eval(lg3, st3); r["prefill8_after_decode"] = sha(lg3, st3)
    lg4, st4 = m.step(mx.array(IDS[None, :1]), st3); mx.eval(lg4, st4); r["decode_after"] = sha(lg4, st4)
    res[os.path.basename(path)] = r; print(os.path.basename(path), {k: v for k, v in r.items() if k not in ("tokens", "generate")}, flush=True)
    del m; mx.clear_cache()
json.dump(res, open(sys.argv[1], "w"))
