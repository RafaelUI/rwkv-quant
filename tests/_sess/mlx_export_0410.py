"""04.10, Mac: схема MLX-файлов MollySophia (affine, группа 64, fp16 scale + bias) на НАШЕМ чекпоинте САМИМ mx.quantize --
коды, scale и bias каждого квантованного тензора кладутся в .npz; оценка -- на сервере (mlx_eval_0410.py), деквант там
побитно равен mx.dequantize (проверено 04.10: q * scale + bias в fp32 -> fp16, 0 отличий на 6 и 4 битах).
Схема (как vs_mlx_affine_2209): int6 -- r/k/v/o, ffn.key/value, эмбеддинг, голова, LoRA «вниз» (w/a/g/v _lora_A) в 6 бит;
int4 -- r/k/v/o и ffn в 4 бита, эмбеддинг, голова, LoRA «вниз» в 6; остальное fp16. 23.09: mx.quantize(fp16(pth)) побитно
равен их опубликованному файлу g1j (41/41 матриц).
    python mlx_export_0410.py <ckpt> <6|4> <out.npz>"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, mlx.core as mx
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK, BM, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t0 = time.time()
M = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16)
out, byt, n = {}, 0.0, 0
def put(name, w, bits):
    global byt, n
    if w is None: return
    a = mx.array(w.float().to(torch.float16).numpy())
    wq, s, b = mx.quantize(a, group_size=64, bits=bits)
    d = np.array(mx.dequantize(wq, s, b, group_size=64, bits=bits))
    wq, s, b = np.array(wq), np.array(s), np.array(b)
    out[name + "|wq"] = wq; out[name + "|s"] = s; out[name + "|b"] = b
    out[name + "|meta"] = np.array([bits, w.shape[0], w.shape[1], int(np.frombuffer(d.tobytes(), dtype=np.uint16).astype(np.uint64).sum() % (1 << 62))], dtype=np.int64)
    byt += w.numel() * (bits + 0.5) / 8; n += 1
put("emb_weight", M.emb_weight, 6); put("head_weight", M.head_weight, 6)
for i, (t, c) in enumerate(zip(M.tmix, M.cmix)):
    for attr in ("r_proj", "k_proj", "v_proj", "o_proj"): put("tmix.%d.%s" % (i, attr), getattr(t, attr), BM)
    for attr in ("w_lora_A", "a_lora_A", "g_lora_A", "v_lora_A"): put("tmix.%d.%s" % (i, attr), getattr(t, attr), 6)
    put("cmix.%d.key" % i, c.key, BM); put("cmix.%d.value" % i, c.value, BM)
total = sum(v.numel() for v in torch.load(CK, map_location="cpu", mmap=True, weights_only=True).values())
qn = sum(int(out[k][1]) * int(out[k][2]) for k in out if k.endswith("|meta"))
est = byt + (total - qn) * 2
out["__info__"] = np.array([BM, n, est], dtype=np.float64)
np.savez(OUT, **out)
print("%s: int%d, квантованных %d, оценка размера %.1f МБ (их формула: (бит + 0.5) / 8 на вес + fp16 прочее), npz %.1f МБ, %.0f с" % (
    os.path.basename(CK), BM, n, est / 1e6, os.path.getsize(OUT) / 1e6, time.time() - t0), flush=True)
