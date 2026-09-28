"""28.09: расширенная калибровка для GPTQ (опыт: на 7.2B/13.3B при 48 окнах корпуса пакета GPTQ ХУЖЕ RTN
на тексте -- en +13.6/+17.7%; корпус пакета -- 48.7 тыс. токенов, en 16%).
Состав (окна по 512, первое окно каждого источника -- разнообразие):
  wiki en/ru/sr -- строки parquet С 20000 (отложенный eval взят из 0..19999, статьи не пересекаются);
  код -- открытые репо (rwkv-quant, rwkv-metal, SwiftRWKV), БЕЗ файлов eval_code_heldout; закрытые не берутся;
  корпус пакета -- все 82 окна (единственный zh).
Порядок -- по кругу между группами (любой префикс сбалансирован). Каждое окно -- проверка 16-грамм против
eval_text_heldout и eval_code_heldout; с пересечением -- отбрасывается. -> ~/Develop/WKV-kvant/gptq_calib_2809.pt
    python make_gptq_calib_2809.py [en=64 ru=48 sr=48 py=40 swift=24]"""
import glob, os, re, sys, collections
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch, pyarrow.parquet as pq
from rwkv_quant.calibration import act_stats as A
Q = dict(en=64, ru=48, sr=48, py=40, swift=24)
for a in sys.argv[1:]:
    k, v = a.split("="); Q[k] = int(v)
L = 512
enc = A._encoder("/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
W = os.path.expanduser("~/Develop/WKV-kvant/")
ev = torch.load(W + "eval_text_heldout.pt", weights_only=False); ec = torch.load(W + "eval_code_heldout.pt", weights_only=False)
G = set()
for t in list(ev["tokens"].tolist()) + list(ec["tokens"].tolist()):
    G.update(tuple(t[i:i + 16]) for i in range(len(t) - 16))
dropped = collections.Counter()
def ok(w, tag):
    if any(tuple(w[i:i + 16]) in G for i in range(len(w) - 16)):
        dropped[tag] += 1; return False
    return True
groups = collections.OrderedDict()
for l in ("en", "ru", "sr"):
    rows = pq.read_table(os.path.expanduser("~/Develop/data/%s.parquet" % l), columns=["text"]).slice(20000, 20000 + 6 * Q[l]).to_pylist()
    out = []
    for r in rows:
        ids = enc(r["text"])
        if len(ids) >= L and ok(ids[:L], l):
            out.append(ids[:L])
        if len(out) >= Q[l]: break
    groups[l] = out
evsrc = set(ec["source"])
for tag, pats in (("py", ["~/Develop/rwkv-quant/rwkv_quant/**/*.py", "~/Develop/rwkv-metal/rwkv_metal/**/*.py"]),
                  ("swift", ["~/Develop/SwiftRWKV/Sources/**/*.swift"])):
    files = sorted(f for p in pats for f in glob.glob(os.path.expanduser(p), recursive=True))
    files = [f for f in files if not any(f.endswith(s.split("/", 1)[1]) and s.split("/", 1)[0] in f for s in evsrc)]
    out = []
    for f in files:
        ids = enc(open(f, encoding="utf-8", errors="ignore").read())
        if len(ids) >= L and ok(ids[:L], tag):
            out.append(ids[:L])
        if len(out) >= Q[tag]: break
    groups[tag] = out
chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", open(A.CORPUS, encoding="utf-8").read()) if c.strip()]
groups["корпус"] = [w[:L] for w in A._windows(chunks, enc, A.SEQ_LEN, 10 ** 7) if len(w) >= L]
order, lab = [], []
it = {k: iter(v) for k, v in groups.items()}
while it:
    for k in list(it):
        w = next(it[k], None)
        if w is None: del it[k]; continue
        order.append(w); lab.append(k)
T = torch.tensor(order, dtype=torch.long)
torch.save(dict(tokens=T, lang=lab, seq_len=L, note="GPTQ-калибровка 28.09: wiki с 20000 строки, открытый код без eval-файлов, корпус пакета"), W + "gptq_calib_2809.pt")
print("окон %d (%d токенов): %s; отброшено по 16-граммам: %s" % (T.shape[0], T.numel(), {k: len(v) for k, v in groups.items()}, dict(dropped)))
