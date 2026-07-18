"""Спекулятивный декодинг с ЖИВЫМ драфтом rwkv7-g1d-0.1b (bf16 dense).
Target: affine6 (1.5B g1h). Greedy. Механика:
- target: pending-трюк из spec_decode_poc.py (state лишь при полном принятии);
- draft: state-снапшот (mx-массивы иммутабельны) -- перед k предложениями
  запоминаем чистый state, после верификации откат бесплатен; в начале
  раунда draft догоняет решённые токены одним T=delta проходом, его же
  логиты дают d1 (первый шаг драфта бесплатен).
Замер: 4 офсета промпта x N_GEN токенов, per-offset и агрегат.
argv: k [n_gen]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/Develop/WKV-kvant"))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.writer import quantize_tensor, _match_group
from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.presets import REDUCTION
from rwkv_quant.calibration.group_config import QuantConfig
from world_tokenizer import RWKV_WORLD_TOKENIZER

TARGET = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
DRAFT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b-ctx8192.pth")
VOCAB = os.path.expanduser("~/Develop/rwkv7-1.5B-g1g-mlx-6bit/rwkv_vocab_v20230424.txt")
TXT = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.txt")
GS = 64; BITS = {"proj": 6, "cmix": 6, "emb_head": 6}
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 128
FLUSH = 3 * K
OFFSETS = (0, 4000, 8000, 12000)

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

print("build target affine6 (1.5B)...", flush=True)
sd = torch.load(TARGET, map_location="cpu")
tensors = {}
for k in list(sd.keys()):
    w = sd.pop(k); g = _match_group(k)
    if g in BITS and w.dim() == 2 and w.shape[1] % GS == 0:
        tensors[k] = emb_roundtrip_qt(k, g, w, BITS[g]) if "emb" in k \
                     else make_affine_qt(k, g, w, BITS[g])
    else:
        tensors[k] = quantize_tensor(k, w, REDUCTION, real_gw=True)
del sd
target = QuantRWKV7(QuantizedCheckpoint(naming="world", n_layer=24, n_embd=2048,
                    head_size=64, vocab_size=65536, tensors=tensors, config_repr="affine"))

print("build draft 0.1b bf16-dense...", flush=True)
dcfg = QuantConfig()   # все группы bits=16 -> dense
dsd = torch.load(DRAFT, map_location="cpu")
dtensors = {k: quantize_tensor(k, w, dcfg) for k, w in dsd.items()}
del dsd
draft = QuantRWKV7(QuantizedCheckpoint(naming="world", n_layer=12, n_embd=768,
                   head_size=64, vocab_size=65536, tensors=dtensors, config_repr="draft"))

tok = RWKV_WORLD_TOKENIZER(VOCAB)
full_text = open(TXT, encoding="utf-8").read()

def am(logits):  # (1,T,V) -> np argmax per position
    return np.array(mx.argmax(logits.astype(mx.float32), axis=-1))[0]

# скорость драфта (compiled step)
st = draft.init_state(1); cur = 100
for _ in range(8):
    lg, st = draft.step(mx.array([[cur]]), st); cur = int(am(lg)[-1])
mx.synchronize(); t0 = time.perf_counter()
for _ in range(32):
    lg, st = draft.step(mx.array([[cur]]), st); cur = int(am(lg)[-1])
mx.synchronize()
print(f"draft compiled step: {(time.perf_counter()-t0)/32*1000:.2f} ms/tok", flush=True)

# бейзлайн target compiled
prompt0 = tok.encode(full_text[:2000])
st = target.init_state(1)
lg, st = target.forward_stateful(mx.array([prompt0]), st, last_only=True)
cur = int(am(lg)[-1])
for _ in range(8):
    lg, st = target.step(mx.array([[cur]]), st); cur = int(am(lg)[-1])
mx.synchronize(); t0 = time.perf_counter()
for _ in range(N_GEN):
    lg, st = target.step(mx.array([[cur]]), st); cur = int(am(lg)[-1])
mx.synchronize()
base_ms = (time.perf_counter()-t0)/N_GEN*1000
print(f"target plain compiled: {base_ms:.2f} ms/tok\n", flush=True)

def spec_run(offset):
    prompt = tok.encode(full_text[offset:offset+2000])
    st_t = target.init_state(1)
    lg, st_t = target.forward_stateful(mx.array([prompt]), st_t, last_only=True)
    first = int(am(lg)[-1])
    st_d = draft.init_state(1)
    _, st_d = draft.forward_stateful(mx.array([prompt]), st_d, last_only=True)
    # st_d покрывает prompt; решённые токены: prompt + [first]
    out = [first]; pending = [first]; delta = [first]   # delta: чего не видел драфт
    rounds = acc_tot = drafted = flushes = 0
    mx.synchronize(); t0 = time.perf_counter()
    while len(out) < N_GEN:
        if len(pending) > FLUSH:
            lg, st_t = target.forward_stateful(mx.array([pending]), st_t, last_only=True)
            nxt = int(am(lg)[-1]); pending = [nxt]; delta.append(nxt); out.append(nxt)
            flushes += 1
            continue
        # драфт догоняет delta одним проходом; его логиты дают d1
        lg_d, st_d = draft.forward_stateful(mx.array([delta]), st_d, last_only=True)
        st_d_clean = st_d
        d = [int(am(lg_d)[-1])]
        dc = d[0]
        for _ in range(K-1):
            lg_d, st_d = draft.step(mx.array([[dc]]), st_d)
            dc = int(am(lg_d)[-1]); d.append(dc)
        st_d = st_d_clean            # откат драфта (предложения не в state)
        # верификация target'ом
        p = len(pending); x = pending + d
        lg, st_new = target.forward_stateful(mx.array([x]), st_t, last_only=False)
        pred = am(lg)
        m = 0
        while m < len(d) and d[m] == int(pred[p-1+m]): m += 1
        bonus = int(pred[p-1+m])
        good = d[:m] + [bonus]
        out.extend(good)
        rounds += 1; acc_tot += m; drafted += len(d)
        if m == len(d):
            st_t = st_new; pending = [bonus]
        else:
            pending = pending + good
        delta = good                  # драфт видел d[:m] лишь как предложения
    mx.synchronize()
    ms = (time.perf_counter()-t0)/len(out)*1000
    return ms, acc_tot, drafted, rounds, flushes, out

print(f"{'офсет':>6s} {'ms/tok':>7s} {'x':>5s} {'приемка':>8s} {'раундов':>7s} {'flush':>5s}")
agg_ms = []; agg_a = agg_d = 0
for off in OFFSETS:
    ms, a, dr, r, f, out = spec_run(off)
    agg_ms.append(ms); agg_a += a; agg_d += dr
    print(f"{off:6d} {ms:7.2f} {base_ms/ms:5.2f} {a}/{dr} ({100*a/max(dr,1):.0f}%) {r:7d} {f:5d}", flush=True)
print(f"\nагрегат: {np.mean(agg_ms):.2f} ms/tok, speedup {base_ms/np.mean(agg_ms):.2f}x, "
      f"приемка {100*agg_a/max(agg_d,1):.0f}%")
print("хвост последнего вывода:", tok.decode(out[-60:])[:200].replace("\n", " "))
