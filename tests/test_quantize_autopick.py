"""Гейт quantize(autopick=True) (24.09). 0.1B g1d, COMPRESSION, бюджет +0.5%.
  1  манифест: у обычного файла ключа autopick НЕТ; у autopick-файла есть, с подписью измерения;
  2  тензоры вне выбора -- ПОБИТНО те же, что в обычном файле; в выборе -- ровно выбранная битность;
  3  размер: прирост к обычному файлу <= бюджет + 0.3% (модель байт select против реального файла);
  4  measure=путь к готовому JSON -> файл побитно тот же, что при measure="auto";
  5  JSON от другого чекпоинта -> отказ (ValueError).
    python tests/test_quantize_autopick.py"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors import safe_open
from rwkv_quant import quantize
from rwkv_quant.calibration import autopick as ap
from rwkv_quant.formats.reader import load_raw
CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
T = "/tmp/x/qa"; os.makedirs(T, exist_ok=True)
B = 0.005
fails = 0
def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print("  %s %s %s" % ("OK  " if ok else "FAIL", name, detail), flush=True)

def man(p):
    with safe_open(p, "pt") as f:
        return json.loads(f.metadata()["rwkvq"])

def blobs(p):
    out = {}
    with safe_open(p, "pt") as f:
        for n in f.keys():
            out.setdefault(n.split("::")[0], hashlib.sha1())
        for n in sorted(f.keys()):
            out[n.split("::")[0]].update(n.encode()); out[n.split("::")[0]].update(f.get_tensor(n).contiguous().view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes())
    return {k: v.hexdigest() for k, v in out.items()}

P, A, A2 = T + "/plain.rwkvq", T + "/auto.rwkvq", T + "/auto_json.rwkvq"
quantize(CK, P, preset="compression", tokenizer=TOK, verbose=False)
quantize(CK, A, preset="compression", tokenizer=TOK, autopick=True, autopick_budget=B, verbose=True)
mp, ma = man(P), man(A)
ov = ma.get("autopick", {}).get("overrides", {})
check("1 обычный файл без autopick", "autopick" not in mp)
check("1 autopick в манифесте", bool(ov) and ma["autopick"]["measure_signature"],
      "подпись %s, вверх %s, вниз %s" % (ma.get("autopick", {}).get("measure_signature"), ma.get("autopick", {}).get("n_up"), ma.get("autopick", {}).get("n_down")))
hp, ha = blobs(P), blobs(A)
same_out = [k for k in hp if k not in ov and hp[k] != ha.get(k)]
check("2 вне выбора побитно те же", not same_out and set(hp) == set(ha), "расходятся: %s" % same_out[:5])
ra = load_raw(A)
wrong = [(k, b, getattr(ra.tensors[k], "bits", 16)) for k, b in ov.items() if (16 if b >= 16 else b) != (getattr(ra.tensors[k], "bits", 16) or 16)]
check("2 выбранные битности в файле", not wrong, str(wrong[:5]))
changed = [k for k in ov if hp.get(k) != ha.get(k)]
check("2 выбранные тензоры действительно изменены", len(changed) == len(ov), "%d из %d" % (len(changed), len(ov)))
d = (os.path.getsize(A) - os.path.getsize(P)) / os.path.getsize(P)
check("3 прирост размера <= бюджет + 0.3%%", d <= B + 0.003, "%+.3f%% (модель %+.3f%%)" % (100 * d, 100 * ma["autopick"]["bytes_frac"]))
mj = os.path.join(ap.MEASURE_CACHE, "measure_%s.json" % ma["autopick"]["measure_signature"])
quantize(CK, A2, preset="compression", tokenizer=TOK, autopick=True, autopick_budget=B, measure=mj, verbose=False)
check("4 measure=путь -> побитно тот же файл", open(A, "rb").read() == open(A2, "rb").read())
bad = json.load(open(mj)); bad["ckpt_sig"] = "0" * 16
json.dump(bad, open(T + "/bad.json", "w"))
try:
    quantize(CK, T + "/bad.rwkvq", preset="compression", tokenizer=TOK, autopick=True, measure=T + "/bad.json", verbose=False)
    check("5 чужой чекпоинт отвергнут", False)
except ValueError as e:
    check("5 чужой чекпоинт отвергнут", True, str(e)[:60])
print("ИТОГ:", "ЗЕЛЁНЫЙ" if fails == 0 else "КРАСНЫЙ (%d)" % fails, flush=True)
sys.exit(1 if fails else 0)
