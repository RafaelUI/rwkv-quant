"""Уровень 2/3 валидации wkv7_infer_var: реальные модели.
usage: test_wkv_var_model.py {ru60m|1.5b}
Decode 64 токенов (после префилла 16) двумя путями: штатный chunked+паддинг
vs прямой wkv7_infer(T=1), сравнение логитов на каждом шаге. Ожидание: побитово 0
(no-op паддинг математически точен, синтетика дала 0.0)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
import rwkv_quant.backends.metal.quant_model as qm

which = sys.argv[1]
if which == "1.5b":
    sd = torch.load(os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"), map_location="cpu")
    meta = dict(naming="world", n_layer=24, n_embd=2048, head_size=64, vocab_size=65536)
else:
    from safetensors.torch import load_file
    sd = load_file(os.path.expanduser("~/Develop/WKV-kvant/ru60m/model.safetensors"))
    meta = dict(naming="custom", n_layer=18, n_embd=448, head_size=64, vocab_size=16000)

cfg = QuantConfig(proj=8, cmix=8, emb_head=8, small=8)
tensors = {k: quantize_tensor(k, w, cfg) for k, w in sd.items()}
del sd
model = qm.QuantRWKV7(QuantizedCheckpoint(tensors=tensors, config_repr="r", **meta))

np.random.seed(0)
prompt = np.random.randint(0, meta["vocab_size"], size=(1, 16), dtype=np.int64)

import importlib
wk = importlib.import_module("rwkv_metal.kernel.wkv7")

def _chunked_ref(r, w, k, v, a, b, state):
    # прежний chunked+padding путь -- сохранён здесь как референс
    B, T, H, S = r.shape
    CH = wk.CHUNK
    outs, pos = [], 0
    while pos < T:
        n = min(CH, T - pos)
        sl = lambda x: x[:, pos:pos+n]
        r_c,w_c,k_c,v_c,a_c,b_c = map(sl,(r,w,k,v,a,b))
        if n < CH:
            pad = CH - n
            z = mx.zeros((B,pad,H,S)); o = mx.ones((B,pad,H,S))
            cat = lambda x, f: mx.concatenate([x, f], axis=1)
            r_c,k_c,v_c,a_c,b_c = [cat(x,z) for x in (r_c,k_c,v_c,a_c,b_c)]
            w_c = cat(w_c,o)
        out_c, state = wk.wkv7_infer(r_c,w_c,k_c,v_c,a_c,b_c,state)
        outs.append(out_c[:, :n]); pos += n
    return mx.concatenate(outs, axis=1), state

def run(use_var, n_decode=64):
    qm._wkv_stateful_impl_backup = getattr(qm, "_wkv_stateful_impl_backup", qm._wkv_stateful)
    qm._wkv_stateful = qm._wkv_stateful_impl_backup if use_var else _chunked_ref
    states = model.init_state(1)
    logits, states = model.forward_stateful(mx.array(prompt), states)
    mx.eval(logits)
    tok = int(mx.argmax(logits[:, -1]).item())
    outs = [np.array(logits[:, -1])]
    toks = [tok]
    for _ in range(n_decode):
        logits, states = model.forward_stateful(mx.array([[tok]]), states)
        mx.eval(logits)
        outs.append(np.array(logits[:, -1]))
        tok = int(mx.argmax(logits[:, -1]).item())
        toks.append(tok)
    return np.stack(outs), toks

t0 = time.perf_counter(); ref, toks_ref = run(False); t_ref = time.perf_counter()-t0
t0 = time.perf_counter(); var, toks_var = run(True);  t_var = time.perf_counter()-t0
qm._wkv_stateful = qm._wkv_stateful_impl_backup

max_abs = np.abs(ref - var).max()
rel = max_abs / (np.abs(ref).max() + 1e-9)
same_toks = toks_ref == toks_var
print(f"[{which}] max_abs={max_abs:.3e} rel={rel:.3e} greedy-токены совпадают: {same_toks}")
print(f"[{which}] время 65 шагов: chunked {t_ref:.2f}s -> var {t_var:.2f}s ({t_ref/t_var:.2f}x)")
ok = max_abs == 0.0 and same_toks
print("[OK]" if ok else "[FAIL] (ожидалось побитовое совпадение)")
sys.exit(0 if ok else 1)
