"""28.09: калибровка v2 из ~/Develop/data/calib_src (fetch_calib_sources_2809) + открытый код (SwiftRWKV, rwkv-metal).
Группы и доли: en 30% (FineWeb-Edu), ru 20% (FineWeb-2 + russian_literature + oasst2 ru), code 20% (UltraData 6 языков +
свой открытый код, без файлов eval), zh 10% (FineWeb-2 + novel_text), sr 10% (кириллица 5 + латиница 5), reason 10%
(qwen3.8 distill + Fable 5.1). Окна по 512 токенов, не больше 2 окон с документа; порядок -- ВЗВЕШЕННЫЙ КРУГ между
группами (любой префикс держит доли); дубликаты и окна с 16-граммами из eval_text_heldout / eval_code_heldout выброшены.
    python make_gptq_calib2_2809.py [N=600]  ->  ~/Develop/WKV-kvant/gptq_calib2_2809.pt {tokens [N,512], lang, src}"""
import collections, glob, json, os, sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
from rwkv_quant.calibration import act_stats as A
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
L, PER_DOC = 512, 2
SRC = os.path.expanduser("~/Develop/data/calib_src")
W = os.path.expanduser("~/Develop/WKV-kvant/")
GROUP = dict(fineweb_edu_en="en", fineweb2_ru="ru", ruslit="ru", oasst2_ru="ru", fineweb2_zh="zh", novel_zh="zh",
             fineweb2_sr_cyrl="sr_cyrl", fineweb2_sr_latn="sr_latn", qwen38_distill="reason", fable51_lite="reason")
SHARE = dict(en=0.30, ru=0.20, code=0.20, zh=0.10, sr_cyrl=0.05, sr_latn=0.05, reason=0.10)
enc = A._encoder("/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
ev = torch.load(W + "eval_text_heldout.pt", weights_only=False); ec = torch.load(W + "eval_code_heldout.pt", weights_only=False)
G16 = set()
for t in ev["tokens"].tolist() + ec["tokens"].tolist():
    G16.update(tuple(t[i:i + 16]) for i in range(len(t) - 16))
pool = collections.defaultdict(list)          # группа -> список списков окон по источникам (для круга внутри группы)
drop = collections.Counter(); seen = set()
def add_doc(group, src, text, bucket):
    ids = enc(text)
    for w in range(min(PER_DOC, len(ids) // L)):
        win = ids[w * L:(w + 1) * L]
        key = tuple(win[:64])
        if key in seen:
            drop["дубль"] += 1; continue
        if any(tuple(win[i:i + 16]) in G16 for i in range(L - 16)):
            drop["16-граммы " + group] += 1; continue
        seen.add(key); bucket.append((win, src))
for f in sorted(glob.glob(SRC + "/*.jsonl")):
    name = os.path.basename(f)[:-6]
    g = "code" if name.startswith("ultradata") else GROUP.get(name)
    if g is None:
        print("пропуск источника без группы:", name); continue
    b = []
    for line in open(f, encoding="utf-8"):
        add_doc(g, name, json.loads(line)["text"], b)
    pool[g].append(b)
evsrc = set(ec["source"])
for tag, pats in (("own_swift", ["~/Develop/SwiftRWKV/Sources/**/*.swift"]), ("own_py", ["~/Develop/rwkv-metal/rwkv_metal/**/*.py", "~/Develop/rwkv-quant/rwkv_quant/**/*.py"])):
    files = sorted(f for p in pats for f in glob.glob(os.path.expanduser(p), recursive=True))
    files = [f for f in files if not any(s.split("/", 1)[0] in f and f.endswith(s.split("/", 1)[1]) for s in evsrc)]
    b = []
    for f in files:
        add_doc("code", tag, open(f, encoding="utf-8", errors="ignore").read(), b)
    pool["code"].append(b)
# внутри группы -- круг по источникам; между группами -- взвешенный круг (наибольший дефицит доли)
streams = {}
for g, lists in pool.items():
    its = [iter(x) for x in lists if x]; out = []
    while its:
        nxt = []
        for it in its:
            v = next(it, None)
            if v is not None:
                out.append(v); nxt.append(it)
        its = nxt
    streams[g] = iter(out)
    print("%-8s окон доступно %d" % (g, len(out)))
order, lab, src = [], [], []
cnt = collections.Counter(); alive = set(streams)
while len(order) < N and alive:
    g = max(alive, key=lambda k: SHARE[k] * (len(order) + 1) - cnt[k])
    v = next(streams[g], None)
    if v is None:
        alive.discard(g); continue
    order.append(v[0]); lab.append(g); src.append(v[1]); cnt[g] += 1
T = torch.tensor(order, dtype=torch.long)
torch.save(dict(tokens=T, lang=lab, src=src, seq_len=L, share=SHARE, note="GPTQ-калибровка v2 28.09 (calib_src + свой открытый код)"), W + "gptq_calib2_2809.pt")
for n in (82, 150, 300, N):
    c = collections.Counter(lab[:n]); print("префикс %4d: %s" % (n, {k: "%.0f%%" % (100 * v / n) for k, v in sorted(c.items())}))
print("окон %d (%d токенов); отброшено: %s" % (T.shape[0], T.numel(), dict(drop)))
