"""Декод: наш .rwkvq против Qwen2.5 MLX 4bit в ОДНОМ процессе, бёрсты ABBA
(закон 1). mlx_lm НЕ ставится в венв (тянет transformers 5.x): модель Qwen2
берётся из распакованного колеса /tmp/mlxlm/pkg, пакет mlx_lm подменён
заглушкой, чтобы не исполнялся его __init__ (он импортирует transformers).
Токены случайные: для скорости greedy-декода текст не важен.
Режимы: sync -- mx.eval на каждом токене; async -- конвейер async_eval, как
generate_step в mlx_lm (CPU строит граф токена t+1, пока GPU считает t).
    python bench_vs_qwen_2209.py <our.rwkvq> <qwen_dir> <ctx> <sync|async> [раундов] [шагов]"""
import json, os, subprocess, sys, time, types
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
pk = types.ModuleType("mlx_lm"); pk.__path__ = ["/tmp/mlxlm/pkg/mlx_lm"]; sys.modules["mlx_lm"] = pk
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models import qwen2
from mlx_lm.models.cache import make_prompt_cache
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

sw = lambda: subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5]
RQ, QD, CTX, MODE = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
R = int(sys.argv[5]) if len(sys.argv) > 5 else 16
N = int(sys.argv[6]) if len(sys.argv) > 6 else 32
s00 = sw()
cfg = json.load(open(os.path.join(QD, "config.json")))
qm = qwen2.Model(qwen2.ModelArgs.from_dict(cfg))
W = mx.load(os.path.join(QD, "model.safetensors"))
W = qm.sanitize(W)
q = cfg["quantization"]
nn.quantize(qm, group_size=q["group_size"], bits=q["bits"], class_predicate=lambda p, m: f"{p}.scales" in W)
qm.load_weights(list(W.items())); mx.eval(qm.parameters()); del W
rm = QuantRWKV7(load_raw(RQ))
print("fast_ln наш=%s  ctx=%d  режим=%s" % (rm.fast_ln, CTX, MODE), flush=True)

rng = np.random.default_rng(0)
# ---- префикс: оба модели прогоняются на CTX случайных токенов (кусками по 512)
qc = make_prompt_cache(qm)
pq = mx.array(rng.integers(0, cfg["vocab_size"], (1, CTX)).astype(np.int32))
pr = mx.array(rng.integers(0, 65536, (1, CTX)).astype(np.int32))
st = rm.init_state(1)
for i in range(0, CTX, 512):
    lq = qm(pq[:, i:i + 512], cache=qc); mx.eval(lq, [c.state for c in qc])
    lr, st = rm.forward_stateful(pr[:, i:i + 512], st, last_only=True); mx.eval(lr, st)
tk = {"Q": mx.argmax(lq[:, -1], -1), "R": mx.argmax(lr[:, -1], -1)}
S = {"R": st}

def stepQ(t):
    return mx.argmax(qm(t[None], cache=qc)[:, -1], -1)
def stepR(t):
    lg, S["R"] = rm.step(t[None], S["R"]); return mx.argmax(lg[:, -1], -1)
STEP = {"Q": stepQ, "R": stepR}

def burst(n, k):
    f, t = STEP[n], tk[n]; t0 = time.perf_counter()
    if MODE == "sync":
        for _ in range(k):
            t = f(t); mx.eval(t)
    else:
        y = f(t); mx.async_eval(y)
        for _ in range(k - 1):
            y2 = f(y); mx.async_eval(y2); mx.eval(y); y = y2
        mx.eval(y); t = y
    tk[n] = t; return (time.perf_counter() - t0) / k * 1000
for _ in range(2):
    burst("R", 8); burst("Q", 8)
s0 = sw(); T = {"R": [], "Q": []}
for r in range(R):
    for n in (("R", "Q") if r % 2 == 0 else ("Q", "R")):
        T[n].append(burst(n, N))
s1 = sw()
a, b = np.array(T["R"]), np.array(T["Q"]); d = b - a
bs = d[rng.integers(0, R, (20000, R))].mean(1)
print("R наш %s: %.2f мс/ток (%.1f ток/с; мин %.2f макс %.2f)" % (os.path.basename(RQ), a.mean(), 1000 / a.mean(), a.min(), a.max()))
print("Q qwen: %.2f мс/ток (%.1f ток/с; мин %.2f макс %.2f)" % (b.mean(), 1000 / b.mean(), b.min(), b.max()))
print("Q-R: %+.3f мс/ток (%+.2f%% от нашего)  95%% CI [%+.3f; %+.3f]  раундов %d x %d" % (
    d.mean(), 100 * d.mean() / a.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), R, N))
print("своп за замер: %s -> %s; за процесс с загрузкой: %s -> %s" % (s0, s1, s00, s1), flush=True)
