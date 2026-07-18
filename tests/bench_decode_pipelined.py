"""Цена хост-синка в decode: eval каждый токен vs eval раз в 8/16
(токен остаётся mx-массивом, argmax на GPU, питон не ждёт). Гейт:
жадная траектория идентична посинковому варианту."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

qm.FUSE = True
model = qm.QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
data = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt"))[:1].numpy()
prompt = mx.array(data[0:1, :64].astype(np.int32))
step = mx.compile(model.forward_stateful)

def decode(n, sync_every, collect=False):
    st = model.init_state(1)
    logits, st = step(prompt, st, True)
    tok = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(tok)
    toks = []
    mx.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        logits, st = step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        if (i + 1) % sync_every == 0:
            mx.eval(tok)
        if collect:
            mx.eval(tok); toks.append(int(tok[0]))
    mx.eval(tok); mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3, toks

t1, ref = decode(48, 1, collect=True)
tp8, got = decode(48, 8, collect=True)   # collect синкает -- только траектория
print(f"траектория sync8 vs sync1: {sum(a==b for a,b in zip(ref,got))}/48")

acc = {1: [], 8: [], 16: []}
for _ in range(5):
    for se in (1, 8, 16):
        acc[se].append(decode(64, se)[0])
for se in (1, 8, 16):
    print(f"eval раз в {se:2d}: {np.median(acc[se]):6.2f} ms/tok")
