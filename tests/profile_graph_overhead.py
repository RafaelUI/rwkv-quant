"""Гипотезы ggml-стиля (граф один раз + фьюз elementwise) на нашем decode:
A) стоимость ПОСТРОЕНИЯ MLX-графа на токен (forward_stateful без eval);
B) mx.compile на реальный шаг (фьюз elementwise-цепочек + кеш графа);
Замеры устойчивые (DVFS)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.presets import REDUCTION
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
import rwkv_quant.backends.metal.quant_model as qm

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
sd = torch.load(CKPT, map_location="cpu")
tensors = {k: quantize_tensor(k, w, REDUCTION) for k, w in sd.items()}
del sd
model = qm.QuantRWKV7(QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                       head_size=64, vocab_size=65536, tensors=tensors, config_repr="r"))
idx = mx.array(np.array([[123]], dtype=np.int64))

def flat(states):
    return [s for st in states for s in st if s is not None]

# --- A: построение графа без eval ---
states = model.init_state(1)
logits, states = model.forward_stateful(idx, states); mx.eval(logits, *flat(states))
t0 = time.perf_counter(); N = 50
for _ in range(N):
    logits, s2 = model.forward_stateful(idx, states)  # без eval
t_build = (time.perf_counter()-t0)/N*1e3
mx.eval(logits, *flat(s2))
print(f"A. построение графа (python, без eval): {t_build:.2f} ms/tok", flush=True)

# --- базовая линия: обычный шаг, устойчиво ---
def spin(step_fn, states, seconds):
    t_end = time.perf_counter() + seconds; n = 0
    while time.perf_counter() < t_end:
        logits, states = step_fn(idx, states)
        mx.eval(logits, *flat(states)); n += 1
    return n, states

_, states = spin(model.forward_stateful, model.init_state(1), 3)
mx.synchronize(); t0 = time.perf_counter()
n, states = spin(model.forward_stateful, states, 5)
mx.synchronize()
t_eager = (time.perf_counter()-t0)/n*1e3
print(f"   eager полный шаг: {t_eager:.2f} ms/tok ({1000/t_eager:.1f} tok/s)", flush=True)

# --- B: mx.compile ---
compiled = mx.compile(model.forward_stateful)
try:
    _, states = spin(compiled, model.init_state(1), 3)
    mx.synchronize(); t0 = time.perf_counter()
    n, states = spin(compiled, states, 5)
    mx.synchronize()
    t_comp = (time.perf_counter()-t0)/n*1e3
    print(f"B. compiled шаг: {t_comp:.2f} ms/tok ({1000/t_comp:.1f} tok/s, {t_eager/t_comp:.2f}x)", flush=True)
    # сверка
    st1 = model.init_state(1); st2 = model.init_state(1)
    l1, _ = model.forward_stateful(idx, st1); l2, _ = compiled(idx, st2)
    mx.eval(l1, l2)
    print(f"   max |diff| eager vs compiled: {float(mx.abs(l1-l2).max()):.3e}", flush=True)
except Exception as e:
    print(f"B. mx.compile не сработал: {type(e).__name__}: {e}", flush=True)
