"""Аблации на COMPILED-пути: каждый вариант = свежий mx.compile
стабнутого forward_stateful (monkeypatch применён ДО компиляции).
A/B-чередование full/ablated. Потолки выигрыша фьюзов."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

model = qm.QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))
N_EMBD = 2048

def bench_fn(fn, n=25, warm=8):
    st = model.init_state(1)
    idx = mx.array(np.array([[123]], dtype=np.int64))
    logits, st = fn(idx, st)
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(warm):
        logits, st = fn(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = fn(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

full_c = mx.compile(model.forward_stateful)

def ab(name, apply, revert, R=5):
    deltas, fulls = [], []
    for _ in range(R):
        tA = bench_fn(full_c)
        apply()
        abl_c = mx.compile(model.forward_stateful)   # свежая трассировка
        tB = bench_fn(abl_c)
        revert()
        fulls.append(tA); deltas.append(tA - tB)
    print(f"{name:26s} delta={np.median(deltas):6.2f} ms (full med {np.median(fulls):5.2f})", flush=True)

print(f"full compiled: {bench_fn(full_c):.2f} ms/tok", flush=True)

_orig_wkv = qm._wkv_stateful
ab("WKV -> passthrough",
   lambda: setattr(qm, "_wkv_stateful", lambda r,w,k,v,a,b,st: (v, st)),
   lambda: setattr(qm, "_wkv_stateful", _orig_wkv))

_saved = []
def lora_off():
    D = N_EMBD
    for blk in model.blocks:
        tm = blk.tmix
        _saved.append((tm.g_lora_A, tm.g_lora_B_w, tm.a_lora_A, tm.a_lora_B_w,
                       tm.w_lora_A, tm.w_lora_B_w,
                       getattr(tm, "v_lora_A", None), getattr(tm, "v_lora_B_w", None)))
        tm.g_lora_A = mx.zeros((1, D)); tm.g_lora_B_w = mx.zeros((D, 1))
        tm.a_lora_A = mx.zeros((1, D)); tm.a_lora_B_w = mx.zeros((D, 1))
        tm.w_lora_A = mx.zeros((1, D)); tm.w_lora_B_w = mx.zeros((D, 1))
        if getattr(tm, "v_lora_A", None) is not None:
            tm.v_lora_A = mx.zeros((1, D)); tm.v_lora_B_w = mx.zeros((D, 1))
def lora_on():
    for blk, s in zip(model.blocks, _saved):
        tm = blk.tmix
        (tm.g_lora_A, tm.g_lora_B_w, tm.a_lora_A, tm.a_lora_B_w,
         tm.w_lora_A, tm.w_lora_B_w, vA, vB) = s
        if vA is not None: tm.v_lora_A, tm.v_lora_B_w = vA, vB
    _saved.clear()
ab("LoRA -> rank-1", lora_off, lora_on)

_saved_cm = []
def cmix_off():
    for blk in model.blocks:
        _saved_cm.append(blk.cmix)
        class ZeroCM:
            def forward_stateful(self, x, ss): return x * 0.0, ss
        blk.cmix = ZeroCM()
def cmix_on():
    for blk, cm in zip(model.blocks, _saved_cm): blk.cmix = cm
    _saved_cm.clear()
ab("cmix -> zero", cmix_off, cmix_on)

_orig_head = model.head
class TinyHead:
    def __call__(self, x): return x[..., :16]
ab("head -> tiny",
   lambda: setattr(model, "head", TinyHead()),
   lambda: setattr(model, "head", _orig_head))
