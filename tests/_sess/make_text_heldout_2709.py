"""Отложенный ТЕКСТ (27.09): статьи Википедии en/ru/sr из ~/Develop/data/*.parquet (владелец),
по 12 окон на язык, 512 токенов с 128-го (мимо шапки). Статьи -- случайные (seed 0) из
первых 20000 строк, длина >= 640 токенов. Проверка: ни одной общей 16-граммы с калибровочным
корпусом (иначе окно выбрасывается). -> ~/Develop/WKV-kvant/eval_text_heldout.pt
Оговорка: Википедия почти наверняка была в обучении модели -- для KL к bf16 это не мешает."""
import os, random, re, sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import pyarrow.parquet as pq
import torch
from rwkv_quant.calibration import act_stats as A
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
enc = A._encoder(TOK)
text = open(A.CORPUS, encoding="utf-8").read()
G = set()
for c in re.split(r"—+ CHUNK —+", text):
    ids = enc(c.strip())
    G.update(tuple(ids[i:i + 16]) for i in range(len(ids) - 16))
wins, lang, src = [], [], []
for l in ("en", "ru", "sr"):
    t = pq.read_table(os.path.expanduser("~/Develop/data/%s.parquet" % l), columns=["title", "text"]).slice(0, 20000).to_pylist()
    rng = random.Random(0); rng.shuffle(t)
    got = 0
    for r in t:
        ids = enc(r["title"] + "\n\n" + r["text"])
        if len(ids) < 640:
            continue
        w = ids[128:640]
        if any(tuple(w[i:i + 16]) in G for i in range(len(w) - 16)):
            print("  пересечение с корпусом, пропуск:", l, r["title"]); continue
        wins.append(w); lang.append(l); src.append(r["title"]); got += 1
        if got == 12:
            break
    print(l, got)
out = os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt")
torch.save(dict(tokens=torch.tensor(wins, dtype=torch.long), lang=lang, source=src, seq_len=512), out)
print("окон %d -> %s" % (len(wins), out)); print(src)
