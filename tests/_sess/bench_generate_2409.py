"""Скорость generate: sync против конвейера async_eval, ABBA в одном процессе (24.09).
Плечи: sync, async (generate_step), sync2/async2 -- контроль A/A; chain_sync/chain_async --
цепочка ровно тех GEMV, что в шаге (как bench_gemv_ceiling_2409), в тех же двух ритмах.
Бёрст = N токенов, продолжая state прошлого бёрста того же плеча.
    RWKVQ_COMP=<путь> python bench_generate_2409.py [раундов] [N]"""
import os, sys, time, subprocess
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import torch
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal import generate as G

COMP = os.environ["RWKVQ_COMP"]
R = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))
sw0 = sw()
raw = load_raw(COMP)
t = raw.tensors["emb.weight"]
TRAF = (os.path.getsize(COMP) - sum(v.numel() * v.element_size() for v in vars(t).values() if isinstance(v, torch.Tensor))) / 1e6
m = QuantRWKV7(raw); del raw, t
P = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))["tokens"][3, :64].tolist()

# цепочка GEMV (как bench_gemv_ceiling_2409)
D = int(m.emb_weight.shape[1]); EMB = m.emb_weight
def chain(tok):
    y = EMB[tok].reshape(1, D).astype(mx.float32)
    for b in m.blocks:
        tm, c = b.tmix, b.cmix
        r = tm._rkv_fused(mx.concatenate([y, y, y], axis=0))[0:1] if tm._rkv_fused is not None else tm.r_proj(y)
        y = tm.o_proj(r * 0.03); h = c.key(y * 0.03); y = c.value(h * 0.001)
    return mx.argmax(m.head(y), axis=-1)
chain_c = mx.compile(chain)

ST = {}
def gen_arm(pipe):
    def f(name):
        st, last = ST.get(name, (None, P))
        out = []
        t0 = time.perf_counter()
        for y, _ in G.generate_step(m, last, N, pipeline=pipe, state=st, out=out):
            pass
        dt = (time.perf_counter() - t0) * 1e3 / N
        ST[name] = (out[0], [int(y.item())])
        return dt
    return f

def chain_arm(pipe):
    def f(name):
        tok = ST.get(name, mx.array([7], dtype=mx.int32))
        t0 = time.perf_counter()
        if not pipe:
            for _ in range(N):
                tok = chain_c(tok); mx.eval(tok)
        else:
            y = chain_c(tok); mx.async_eval(y)
            for _ in range(N - 1):
                y2 = chain_c(y); mx.async_eval(y2); mx.eval(y); y = y2
            mx.eval(y); tok = y
        dt = (time.perf_counter() - t0) * 1e3 / N
        ST[name] = tok
        return dt
    return f

ARMS = {"sync": gen_arm(False), "async": gen_arm(True), "sync2": gen_arm(False), "async2": gen_arm(True),
        "chain_sync": chain_arm(False), "chain_async": chain_arm(True)}
names = list(ARMS)
for _ in range(2):
    for n in names:
        ARMS[n](n)
sw1 = sw()
V = {n: [] for n in names}
for r in range(R):
    for n in (names if r % 2 == 0 else names[::-1]):
        V[n].append(ARMS[n](n))
sw2 = sw()
rng = np.random.default_rng(0)
def ci(a, b):
    d = np.array(V[b]) - np.array(V[a])
    bs = d[rng.integers(0, R, (20000, R))].mean(1)
    return d.mean(), 100 * d.mean() / np.mean(V[a]), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
print("=== %s, трафик %.1f МБ, раундов %d x %d ток ===" % (os.path.basename(COMP), TRAF, R, N))
for n in names:
    a = np.array(V[n]); med = float(np.median(a))
    gb = TRAF if not n.startswith("chain") else float("nan")
    print("  %-11s медиана %7.3f мс/ток (%6.2f т/с) разброс %4.1f%%  полоса %s" % (
        n, med, 1000 / med, 100 * (a.max() - a.min()) / med,
        "%.1f ГБ/с = %.1f%% от 104" % (TRAF / med, 100 * TRAF / med / 104) if not n.startswith("chain") else "--"))
for a, b in (("sync", "sync2"), ("async", "async2"), ("sync", "async"), ("chain_sync", "chain_async"), ("async", "chain_async")):
    d, p, lo, hi = ci(a, b)
    print("  %-11s -> %-11s %+7.3f мс/ток (%+6.2f%%)  95%% CI [%+.3f; %+.3f]" % (a, b, d, p, lo, hi))
print("  своп: за прогрев %+.1f МБ, за замер %+.1f МБ" % (sw1 - sw0, sw2 - sw1), flush=True)
