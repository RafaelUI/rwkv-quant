"""Отложенный набор КОДА (24.09, решение владельца): Python/Swift из его git-репозиториев,
без файлов калибровочного корпуса (проверка и по пути, и по содержимому: начало каждого
кодового чанка корпуса не должно встречаться в файле). Axon и Synapse закрыты -- набор
ТОЛЬКО для локального замера: лежит в WKV-kvant, НЕ в репозитории и НЕ в пакете.
Окно = 512 токенов с 256-го (мимо шапки импортов), файлы < 768 токенов пропускаются;
выбор файлов детерминирован (сортировка по sha1 пути).
    python make_code_heldout_2409.py -> ~/Develop/WKV-kvant/eval_code_heldout.pt"""
import hashlib, os, re, subprocess, sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
from rwkv_quant.calibration import act_stats as A, autopick as ap
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
DV = os.path.expanduser("~/Develop")
PLAN = [("rwkv-quant", "py", "rwkv_quant/", 4), ("rwkv-metal", "py", "rwkv_metal/", 4), ("EmbeddingRWKV", "py", "", 4),
        ("SwiftRWKV", "swift", "", 3), ("Axon", "swift", "", 3), ("Synapse", "swift", "", 3),
        ("Neuro", "swift", "", 1), ("Impulse", "swift", "", 2)]
EXCL = {"rwkv-metal/rwkv_metal/kernel/wkv7.py", "Impulse/Structure/ProjectSearchView.swift", "Neuro/Neuro/RemoteCatalog.swift"}
enc = A._encoder(TOK)
text = open(A.CORPUS, encoding="utf-8").read()
code_heads = [c.strip()[:300] for c in re.split(r"—+ CHUNK —+", text) if c.strip() and ap._lang(c.strip()) == "code"]
assert len(code_heads) == 3, len(code_heads)
wins, lang, src = [], [], []
for repo, ext, prefix, k in PLAN:
    files = subprocess.run(["git", "-C", os.path.join(DV, repo), "ls-files", "*." + ext], capture_output=True, text=True).stdout.split()
    files = sorted((f for f in files if f.startswith(prefix)), key=lambda f: hashlib.sha1((repo + "/" + f).encode()).hexdigest())
    got = 0
    for f in files:
        rel = repo + "/" + f
        if rel in EXCL:
            continue
        try:
            t = open(os.path.join(DV, rel), encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if any(h[:200] in t for h in code_heads):
            print("  исключён по содержимому:", rel); continue
        ids = enc(t)
        if len(ids) < 768:
            continue
        wins.append(ids[256:768]); lang.append(ext); src.append(rel); got += 1
        if got == k:
            break
    print("%-14s %-5s %d из %d" % (repo, ext, got, k))
out = os.path.expanduser("~/Develop/WKV-kvant/eval_code_heldout.pt")
torch.save(dict(tokens=torch.tensor(wins, dtype=torch.long), lang=lang, source=src, seq_len=512,
                note="ЛОКАЛЬНЫЙ замер; Axon/Synapse закрыты -- не публиковать, не класть в корпус"), out)
print("окон %d -> %s" % (len(wins), out))
for s_, l in zip(src, lang):
    print("  ", l, s_)
