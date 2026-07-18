"""Спекулятивный декодинг (greedy) на ЧЕМПИОНЕ (gw sb6, /tmp/champion_v2.rwkvq)\nпосле N-батчевого кернеля. Вариант spec_decode_poc.py.

Механика без правок model-кода:
- state S покрывает префикс; pending -- решённые токены, ещё не в S;
- раунд: x = pending + draft(k) одним forward_stateful;
  * позиции pending всегда валидны;
  * драфт принимается по префиксу argmax-совпадений + бонус-токен;
  * state S' берём ТОЛЬКО при полном принятии драфта, иначе S старый,
    а принятое уходит в pending (догонит state в следующем проходе);
  * если pending распух (> FLUSH) -- чистый advance-проход.
- драфт: n-gram lookup по истории (prompt+выход), без драфт-модели.

Замеры в одном процессе: plain eager, plain compiled step, spec-цикл.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/Develop/WKV-kvant"))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor, _match_group
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.presets import REDUCTION
from world_tokenizer import RWKV_WORLD_TOKENIZER

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
VOCAB = os.path.expanduser("~/Develop/rwkv7-1.5B-g1g-mlx-6bit/rwkv_vocab_v20230424.txt")
TXT = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.txt")
GS = 64; BITS = {"proj": 6, "cmix": 6, "emb_head": 6}
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_GEN = 256
FLUSH = 2 * K

def make_affine_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits); mx.eval(wq, s, b)
    qt = QuantizedTensor(key=key, group=group, bits=bits, shape=tuple(w.shape))
    qt.gw_mode = "mlx_affine"; qt.mlx_weight = wq; qt.mlx_scales = s
    qt.mlx_biases = b; qt.mlx_group_size = GS; qt.mlx_bits = bits
    return qt

def emb_roundtrip_qt(key, group, w, bits):
    wh = mx.array(w.float().numpy()).astype(mx.float16)
    wq, s, b = mx.quantize(wh, group_size=GS, bits=bits)
    deq = mx.dequantize(wq, scales=s, biases=b, group_size=GS, bits=bits); mx.eval(deq)
    dense = torch.from_numpy(np.array(deq.astype(mx.float32))).to(torch.bfloat16)
    return QuantizedTensor(key=key, group=group, bits=16, shape=tuple(w.shape), dense=dense)

from rwkv_quant.formats.reader import load_raw
model = QuantRWKV7(load_raw("/tmp/champion_v2.rwkvq"))

tok = RWKV_WORLD_TOKENIZER(VOCAB)
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 0
prompt = tok.encode(open(TXT, encoding="utf-8").read()[OFF:OFF+2000])
print(f"prompt: {len(prompt)} токенов", flush=True)

def argmax_np(logits):  # logits (1,T,V) -> np int per position
    return np.array(mx.argmax(logits.astype(mx.float32), axis=-1))[0]

def prefill(tokens):
    st = model.init_state(1)
    lg, st = model.forward_stateful(mx.array([tokens]), st, last_only=True)
    first = int(argmax_np(lg)[-1])
    return st, first

# ---------- бейзлайн 1: plain eager ----------
st, t0tok = prefill(prompt)
out_plain = [t0tok]
mx.synchronize(); t0 = time.perf_counter()
cur = t0tok
for _ in range(N_GEN - 1):
    lg, st = model.forward_stateful(mx.array([[cur]]), st, last_only=True)
    cur = int(argmax_np(lg)[-1]); out_plain.append(cur)
mx.synchronize()
dt_eager = (time.perf_counter()-t0)/(N_GEN-1)*1000
print(f"plain eager:    {dt_eager:6.2f} ms/tok", flush=True)

# ---------- бейзлайн 2: compiled step ----------
st, cur = prefill(prompt)
for _ in range(8):
    lg, st = model.step(mx.array([[cur]]), st); cur = int(argmax_np(lg)[-1])
mx.synchronize(); t0 = time.perf_counter()
for _ in range(N_GEN):
    lg, st = model.step(mx.array([[cur]]), st); cur = int(argmax_np(lg)[-1])
mx.synchronize()
dt_step = (time.perf_counter()-t0)/N_GEN*1000
print(f"plain compiled: {dt_step:6.2f} ms/tok", flush=True)

# ---------- n-gram драфт ----------
def ngram_draft(hist, k, nmax=8, nmin=2):
    H = len(hist)
    for n in range(min(nmax, H-1), nmin-1, -1):
        suf = hist[-n:]
        # ищем последнее раннее вхождение suf
        for s in range(H - n - 1, -1, -1):
            if hist[s:s+n] == suf:
                nxt = hist[s+n:s+n+k]
                if nxt: return nxt
    return []

# ---------- спекулятивный цикл ----------
st, cur = prefill(prompt)
_warm_st = model.init_state(1)
_, _warm_st = model.forward_stateful(mx.array([prompt[:64]]), _warm_st, last_only=True)
for _T in range(1, FLUSH + K + 2):
    _lg, _ = model.step(mx.array([[100] * _T]), _warm_st)
    mx.eval(_lg)
mx.synchronize()
hist = list(prompt) + [cur]
out_spec = [cur]
pending = [cur]          # cur ещё не в state
rounds = accepted_total = drafted_total = flushes = 0
mx.synchronize(); t0 = time.perf_counter()
while len(out_spec) < N_GEN:
    if len(pending) > FLUSH:
        lg, st = model.step(mx.array([pending]), st)
        nxt = int(argmax_np(lg)[-1])
        pending = [nxt]; hist.append(nxt); out_spec.append(nxt)
        flushes += 1
        continue
    draft = ngram_draft(hist, K)
    p = len(pending)
    x = pending + draft
    lg, st_new = model.step(mx.array([x]), st)
    pred = argmax_np(lg)          # pred[i] = argmax после x[i]
    m = 0
    while m < len(draft) and draft[m] == int(pred[p-1+m]):
        m += 1
    bonus = int(pred[p-1+m])
    good = draft[:m] + [bonus]
    out_spec.extend(good); hist.extend(good)
    rounds += 1; accepted_total += m; drafted_total += len(draft)
    if m == len(draft) and len(draft) > 0:
        st = st_new                 # state покрывает весь x
        pending = [bonus]
    else:
        pending = pending + good    # state старый, всё принятое -- в хвост
mx.synchronize()
dt_spec = (time.perf_counter()-t0)/len(out_spec)*1000
acc = accepted_total/max(drafted_total,1)
print(f"spec (k={K}):     {dt_spec:6.2f} ms/tok  | приняно {accepted_total}/{drafted_total} "
      f"({100*acc:.0f}%), раундов {rounds}, flush {flushes}", flush=True)
print(f"speedup vs compiled: {dt_step/dt_spec:.2f}x", flush=True)

# идентичность выходов (greedy): plain vs spec могут расходиться из-за
# fp16-чанкинга T>1 -- сообщаем долю совпадения, это диагностика, не гейт
same = sum(a == b for a, b in zip(out_plain, out_spec)) / min(len(out_plain), len(out_spec))
print(f"совпадение plain/spec первых токенов: {100*same:.0f}%")
print("---- вывод spec (хвост): ----")
print(tok.decode(out_spec[-120:]))
