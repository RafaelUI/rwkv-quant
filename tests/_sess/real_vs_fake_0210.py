"""02.10: откуда расхождение реального пути (Metal) с серверным fake-путём у COMPRESSION (1.5B код: +1.82 против +1.49%).
Третье плечо: ДЕКВАНТ ФАЙЛА в torch -- все тензоры файла через reader.load_dequantized подставляются в state dict плотного
чекпоинта, и RWKV7Ref (fp32-счёт, cfg=None -- без всякого fake-квантования) считает CE на тех же окнах.
  файл-в-torch ~ сервер  -> расходятся КЕРНЕЛИ/рантайм Metal;
  файл-в-torch ~ Metal   -> расходится сам ФАЙЛ с fake-путём (что-то квантуется иначе).
    python real_vs_fake_0210.py <файл.rwkvq> <ckpt> <метка> <плечо>   -> добавляет плечо "<плечо>_torch" в real_ppl_heldout_0110.json"""
import math, os, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant"); sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests/_sess")
import torch
import real_ppl_heldout_0110 as R
from rwkv_quant.formats.reader import load_dequantized
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
F, CK, LAB, ARM = sys.argv[1:5]
t0 = time.time(); s0 = R.swap()
sd = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)
dq = load_dequantized(F)
out, n_q, n_t, miss = {}, 0, 0, []
for k, v in sd.items():
    if k in dq:
        w = dq[k]
        if tuple(w.shape) != tuple(v.shape):
            if w.dim() == 2 and tuple(w.T.shape) == tuple(v.shape):
                w = w.T; n_t += 1
            elif w.numel() == v.numel():
                w = w.reshape(v.shape)
            else:
                raise ValueError("форма %s: файл %s, чекпоинт %s" % (k, tuple(w.shape), tuple(v.shape)))
        out[k] = w.to(v.dtype).contiguous(); n_q += 1
    else:
        out[k] = v; miss.append(k)
extra = sorted(set(dq) - set(sd))
print("тензоров из файла %d (транспонировано %d), только из чекпоинта %d %s, только в файле %d %s" % (
    n_q, n_t, len(miss), miss[:4], len(extra), extra[:4]), flush=True)
tmp = "/tmp/deq_%s_%s.pth" % (LAB, ARM)
torch.save(out, tmp); del out, dq, sd
data, langs, kind = R.windows()
M = RWKV7Ref(tmp, device="mps", dtype=torch.bfloat16, compute_dtype=torch.float32)
ce = []
with torch.no_grad():
    for i in range(0, len(data), 4):
        X = torch.tensor(data[i:i + 4, :-1], device="mps"); Y = torch.tensor(data[i:i + 4, 1:], device="mps")
        lp = torch.log_softmax(M.forward(X).float(), -1)
        ce += [float(v) for v in -lp.gather(-1, Y[..., None])[..., 0].mean(-1).cpu()]
os.remove(tmp)
R.save(LAB, ARM + "_torch", ce, dict(file=os.path.basename(F), bytes=os.path.getsize(F), langs=langs, kind=kind,
                                    seconds=round(time.time() - t0), swap_before=s0, swap_after=R.swap()))
import numpy as np
print("%s %s_torch: ppl текст %.4f код %.4f, %.0f с, своп %s -> %s" % (LAB, ARM, math.exp(np.mean([c for c, k in zip(ce, kind) if k == "text"])),
      math.exp(np.mean([c for c, k in zip(ce, kind) if k == "code"])), time.time() - t0, s0, R.swap()), flush=True)
