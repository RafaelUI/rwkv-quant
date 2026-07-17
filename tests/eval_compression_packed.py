"""E2E COMPRESSION с INT4 bit-packing: размер .rwkvq + decode tok/s.
ppl проверен отдельно: tests/diagnose_one.py compression_fixed -> 16.9491
(задокументировано +48.3% ~= 16.95) -- packing побитово воспроизводит codes.
Замер decode -- устойчивый (см. DVFS-примечание в test_int4_packing.py)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.presets import COMPRESSION
from rwkv_quant.formats.writer import save
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
OUT_Q = "/tmp/compression_packed.rwkvq"

# Переиспользуем готовый .rwkvq: полная переквантовка 1.5B раздувает RSS
# (torch-копии) и через page-outs портит последующий замер decode -- две
# "аномалии" 27мс против честных 16 были именно этим.
if not os.path.exists(OUT_Q):
    sd = torch.load(CKPT, map_location="cpu")
    t0 = time.time()
    save(sd, COMPRESSION, OUT_Q, naming="world", n_layer=24, n_embd=2048,
         head_size=64, vocab_size=65536)
    del sd
    print(f"квантовка: {time.time()-t0:.0f}s", flush=True)
print(f"файл: {os.path.getsize(OUT_Q)/1e6:.0f} MB", flush=True)

ckpt = load_raw(OUT_Q)
n_packed = sum(1 for t in ckpt.tensors.values() if t.codes_packed is not None)
n_int8 = sum(1 for t in ckpt.tensors.values() if t.codes is not None)
print(f"тензоров: packed {n_packed}, int8 {n_int8}", flush=True)

model = qm.QuantRWKV7(ckpt)
states = model.init_state(1)
idx = mx.array(np.array([[123]], dtype=np.int64))

def _flat(st):
    return [s for x in st for s in x if s is not None]

def spin(seconds):
    global states
    t_end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < t_end:
        logits, states = model.step(idx, states)
        # ВАЖНО: eval'ить logits И states. Ленивые state-ноды между
        # итерациями ломают конвейер (compiled-шаг материализует входы
        # синхронно): наблюдалось 26-27 мс/ток вместо честных 16.
        mx.eval(logits, *_flat(states))
        n += 1
    return n

spin(4)  # прогрев + DVFS
mx.synchronize(); t0 = time.perf_counter()
n = spin(6)
mx.synchronize()
ms = (time.perf_counter() - t0) / n * 1e3
print(f"decode: {ms:.2f} ms/tok ({1000/ms:.2f} tok/s, {n} токенов)", flush=True)
