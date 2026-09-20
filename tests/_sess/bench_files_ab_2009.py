"""Декод двух .rwkvq в ОДНОМ процессе, бёрсты чередуются ABBA (закон 1:
между процессами мс/ток несравнимы). Парная разность по раундам,
бутстрэп; своп до и после.
    python bench_files_ab_2009.py <A.rwkvq> <B.rwkvq> [раундов] [шагов]"""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, torch
import mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

sw = lambda: subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5]
A, B = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 16
N = int(sys.argv[4]) if len(sys.argv) > 4 else 32
M = {"A": QuantRWKV7(load_raw(A)), "B": QuantRWKV7(load_raw(B))}
data = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))["tokens"][:1].numpy()
p = mx.array(data[:, :64].astype(np.int32))
st, tk = {}, {}
for n, m in M.items():
    lg, st[n] = m.forward_stateful(p, m.init_state(1), last_only=True)
    tk[n] = mx.argmax(lg[:, -1], axis=-1)
def burst(n, k):
    m = M[n]; t0 = time.perf_counter()
    for _ in range(k):
        lg, st[n] = m.step(tk[n][None], st[n]); tk[n] = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(tk[n]); return (time.perf_counter() - t0) / k * 1000
for _ in range(2):
    burst("A", 8); burst("B", 8)
s0 = sw(); T = {"A": [], "B": []}
for r in range(R):
    for n in (("A", "B") if r % 2 == 0 else ("B", "A")):
        T[n].append(burst(n, N))
s1 = sw()
a, b = np.array(T["A"]), np.array(T["B"]); d = b - a
rng = np.random.default_rng(0); bs = d[rng.integers(0, R, (20000, R))].mean(1)
print("A %s: %.2f мс/ток (мин %.2f макс %.2f)" % (os.path.basename(A), a.mean(), a.min(), a.max()))
print("B %s: %.2f мс/ток (мин %.2f макс %.2f)" % (os.path.basename(B), b.mean(), b.min(), b.max()))
print("B-A: %+.3f мс/ток (%+.2f%%)  95%% CI [%+.3f; %+.3f]  раундов %d x %d шагов" % (
    d.mean(), 100 * d.mean() / a.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), R, N))
print("своп: до %s, после %s" % (s0, s1))
