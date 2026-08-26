# -*- coding: utf-8 -*-
"""Эталон логитов в fp32 (активации fp32, веса -- оригинальные bf16).

Нынешний эталон KL снят в bf16, и на 0.1B измерено, что он отстоит от
fp32 на 0.000756 нат/токен, тогда как MLX-путь -- на 0.000004. То есть
эталон вносит собственную ошибку того же порядка, что измеряемый эффект.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, torch
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT = os.environ["RWKVQ_CKPT"]
OUT = os.environ["RWKVQ_REF32"]
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
NSEQ, SEQLEN = 8, 512

data = torch.load(CORPUS)["tokens"][:NSEQ, :SEQLEN]
m = RWKV7Ref(CKPT, device="cpu", dtype=torch.float32)
mm = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32,
                               shape=(NSEQ, SEQLEN - 1, 65536))
t0 = time.time()
with torch.no_grad():
    for i in range(NSEQ):
        lg = m.forward(data[i:i + 1, :-1])
        out = lg[0].float().numpy()
        if not np.isfinite(out).all() or float(np.abs(out).max()) == 0.0:
            raise RuntimeError("логиты %d вырождены (закон 21)" % i)
        mm[i] = out
        print("  %d/%d  %.0f с" % (i + 1, NSEQ, time.time() - t0), flush=True)
        del lg, out
mm.flush()
print("готово: %s за %.0f с" % (OUT, time.time() - t0))
