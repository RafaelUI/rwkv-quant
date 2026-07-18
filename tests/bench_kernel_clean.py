"""Чистая кернельная полоса (синк амортизирован): head x8 за один eval;
tmix 96 GEMV dep vs indep; цена хост-синка (пустая цепочка).
Кеш-эффекты: head 94MB >> SLC, tmix циклом по 24 слоям (70MB+)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
head = model.head
tmix_lins = [l for b in model.blocks
             for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj)]
x = mx.array(np.random.randn(1, 2048).astype(np.float32)); mx.eval(x)

MB_HEAD = (head.codes.size + head.qs.size + head.qm.size + head.qh.size
           + head.d.size*2 + head.dm.size*2) / 1e6
def szall(lins):
    s = 0
    for l in lins:
        s += (l.codes.size + l.qs.size + l.qm.size + l.d.size*2 + l.dm.size*2
              + (l.qh.size if l.has_qh else 0))
    return s / 1e6
MB_TMIX = szall(tmix_lins)

def bench(fn, reps=10, warm=3):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.median(ts)

def head_x8(): return [head(x) for _ in range(8)]
def tmix_dep():
    y = x
    for l in tmix_lins: y = l(y)
    return [y]
def tmix_indep(): return [l(x) for l in tmix_lins]
def sync_only(): return [x + 1.0]

acc = {n: [] for n in ("head8", "t_dep", "t_ind", "sync")}
for _ in range(5):
    acc["head8"].append(bench(head_x8))
    acc["t_dep"].append(bench(tmix_dep))
    acc["t_ind"].append(bench(tmix_indep))
    acc["sync"].append(bench(sync_only, reps=30))

sync = np.median(acc["sync"])
h8 = np.median(acc["head8"]); td = np.median(acc["t_dep"]); ti = np.median(acc["t_ind"])
print(f"хост-синк (пустой eval):    {sync:6.3f} ms")
print(f"head x8 (за 1 eval):        {h8:7.3f} ms -> {(h8-sync)/8:6.3f} ms/GEMV = {MB_HEAD/((h8-sync)/8):5.1f} GB/s")
print(f"tmix 96 dep:                {td:7.3f} ms -> {MB_TMIX/(td-sync):5.1f} GB/s")
print(f"tmix 96 indep:              {ti:7.3f} ms -> {MB_TMIX/(ti-sync):5.1f} GB/s  (щели deps = {td-ti:+5.2f} ms)")
