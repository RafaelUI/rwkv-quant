"""03.10: ОТКРЫТЫЙ отложенный набор кода для статьи (раздел 15, п. 2): только открытые репозитории владельца
(rwkv-quant, rwkv-metal, SwiftRWKV, Impulse); Axon / Synapse / Neuro / EmbeddingRWKV не берутся.
Состав: (а) окна прежнего eval_code_heldout из открытых репозиториев -- как есть (общая часть со старым набором);
(б) новые окна из тех же репозиториев, содержимое -- из git HEAD (sha в файле), один файл -- одно окно
(512 токенов с 256-го, файлы < 768 токенов пропускаются), порядок файлов -- по sha1 пути, репозитории по кругу.
Каждое окно: 0 общих 16-грамм с калибровкой GPTQ (gptq_calib2_2809, gptq_calib_2809), с calib_corpus пакета
и с уже взятыми окнами. Цель -- 24 py + 24 swift.
    python make_code_open_0310.py [--dry] -> ~/Develop/WKV-kvant/eval_code_open.pt"""
import hashlib, os, subprocess, sys, collections
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
from rwkv_quant.calibration import act_stats as A
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
DV = os.path.expanduser("~/Develop"); W = os.path.expanduser("~/Develop/WKV-kvant/")
OPEN = ("rwkv-quant", "rwkv-metal", "SwiftRWKV", "Impulse")
PLAN = {"py": [("rwkv-quant", ""), ("rwkv-metal", "")], "swift": [("SwiftRWKV", ""), ("Impulse", "")]}
NEED = {"py": 24, "swift": 24}
L, N = 512, 16
enc = A._encoder(TOK)
def grams(t):
    return {tuple(t[i:i + N]) for i in range(len(t) - N + 1)}
G = set(); nsrc = collections.OrderedDict()
for f in ("gptq_calib2_2809.pt", "gptq_calib_2809.pt"):
    d = torch.load(W + f, weights_only=False); toks = d["tokens"].tolist() if isinstance(d, dict) else d.tolist()
    for t in toks: G |= grams(t)
    nsrc[f] = len(toks)
cc = enc(open(A.CORPUS, encoding="utf-8").read()); G |= grams(cc); nsrc["calib_corpus"] = len(cc)
jl = os.path.join(DV, "rwkv-quant/rwkv_quant/data/gptq_calib.jsonl")
if os.path.exists(jl):
    import json
    k = 0
    for line in open(jl, encoding="utf-8"):
        o = json.loads(line); t = enc(o["text"] if isinstance(o, dict) else o); G |= grams(t); k += 1
    nsrc["gptq_calib.jsonl"] = k
print("калибровка:", dict(nsrc), "16-грамм", len(G))
old = torch.load(W + "eval_code_heldout.pt", weights_only=False)
wins, lang, src, origin = [], [], [], []
mine = set()
for t, l, s in zip(old["tokens"].tolist(), old["lang"], old["source"]):
    if s.split("/")[0] not in OPEN:
        continue
    g = grams(t); assert not (g & G), ("старое окно пересекается с калибровкой", s)
    wins.append(t); lang.append(l); src.append(s); origin.append("heldout_2409"); mine |= g
print("из прежнего набора (открытые):", collections.Counter(lang), "закрытых отброшено", len(old["source"]) - len(wins))
sha = {r: subprocess.run(["git", "-C", os.path.join(DV, r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() for r in OPEN}
dirty = {r: subprocess.run(["git", "-C", os.path.join(DV, r), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() != "" for r in OPEN}
print("HEAD:", sha, "грязные:", dirty)
drop = collections.Counter()
for ext, repos in PLAN.items():
    cands = []
    for repo, prefix in repos:
        fl = subprocess.run(["git", "-C", os.path.join(DV, repo), "ls-files", "*." + ext], capture_output=True, text=True).stdout.split("\n")
        fl = sorted((f for f in fl if f and f.startswith(prefix) and "/_sess/" not in f), key=lambda f: hashlib.sha1((repo + "/" + f).encode()).hexdigest())
        cands.append([(repo, f) for f in fl])
    order = [x for tup in zip(*[c + [None] * (max(map(len, cands)) - len(c)) for c in cands]) for x in tup if x]
    have = sum(1 for l in lang if l == ext)
    for repo, f in order:
        if have >= NEED[ext]: break
        rel = repo + "/" + f
        if rel in src: continue
        r = subprocess.run(["git", "-C", os.path.join(DV, repo), "show", "HEAD:" + f], capture_output=True)
        try: t = r.stdout.decode("utf-8")
        except UnicodeDecodeError: drop["не utf-8"] += 1; continue
        ids = enc(t)
        if len(ids) < 768: drop[ext + " короткий"] += 1; continue
        w = ids[256:768]; g = grams(w)
        if g & G: drop[ext + " в калибровке"] += 1; continue
        if g & mine: drop[ext + " дубль окна"] += 1; continue
        wins.append(w); lang.append(ext); src.append(rel); origin.append("HEAD"); mine |= g; have += 1
    print("%-5s %d из %d (кандидатов %d)" % (ext, have, NEED[ext], len(order)))
print("отброшено:", dict(drop)); print("по репозиториям:", collections.Counter((s.split("/")[0], l) for s, l in zip(src, lang)))
assert all(not (grams(w) & G) for w in wins)
if "--dry" not in sys.argv:
    out = W + "eval_code_open.pt"
    torch.save(dict(tokens=torch.tensor(wins, dtype=torch.long), lang=lang, source=src, origin=origin, seq_len=L, head_sha=sha,
                    note="ОТКРЫТЫЙ набор кода: только открытые репозитории владельца; 0 общих 16-грамм с калибровкой"), out)
    print("окон %d -> %s" % (len(wins), out))
for s_, l, o in zip(src, lang, origin): print("  ", l, o, s_)
