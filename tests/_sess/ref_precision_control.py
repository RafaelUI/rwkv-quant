# -*- coding: utf-8 -*-
"""ЧЕЙ ЭТО ПОЛ: ПУТИ ИЛИ ЭТАЛОНА (0.1B, дёшево).

Весь KL проекта меряет расстояние до RWKV7Ref в bf16. Реальный MLX-путь
отстоит от него на 0.000575 нат/токен ПРИ НУЛЕВОМ КВАНТОВАНИИ (1.5B), и
это записывалось бы как "цена пути". Но у эталона есть СВОЯ ошибка, и
знак разницы из одного расстояния не следует: MLX-путь может быть не
хуже эталона, а ТОЧНЕЕ него.

Строим независимый эталон в fp32 и меряем расстояние до него у обоих:

    KL(fp32 ‖ torch-bf16)   -- ошибка НАШЕГО ЭТАЛОНА
    KL(fp32 ‖ mlx-bf16path) -- ошибка MLX-ПУТИ

Если второе меньше первого, то "пол пути" -- артефакт выбора эталона, и
все KL проекта завышены на эту величину неизвестным образом.

0.1B выбран сознательно: вопрос качественный (кто ближе), а fp32-прогон
1.5B стоил бы 6.1 ГБ. Закон 10 обязывает перепроверить на масштабе, если
ответ окажется решающим.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch

CKPT = "/Users/s/rwkv7-g1d-0.1b-ctx8192.pth"
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
NSEQ, SEQLEN = 8, 512
REF32 = "/tmp/ref32_0p1b_ml_8x512.npy"

from ablate_subgroups import kl_stats, boot_ci   # noqa: E402

data = torch.load(CORPUS)["tokens"][:NSEQ, :SEQLEN]


def torch_logits(dtype, device):
    from rwkv_quant.models.rwkv7_ref import RWKV7Ref
    m = RWKV7Ref(CKPT, device=device, dtype=dtype)
    for i in range(NSEQ):
        lg = m.forward(data[i:i + 1, :-1].to(device))
        yield lg[0].float().cpu().numpy()
        del lg
    del m


def mlx_logits():
    import mlx.core as mx
    from rwkv_quant.formats.schema import QuantizedCheckpoint, QuantizedTensor
    from rwkv_quant.backends.metal.quant_model import QuantRWKV7
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    r_k = next(v for k, v in sd.items() if k.endswith("r_k"))
    meta = dict(naming="world", n_layer=n_layer,
                n_embd=int(sd["emb.weight"].shape[1]),
                vocab_size=int(sd["emb.weight"].shape[0]),
                head_size=int(r_k.shape[-1]))
    t = {k: QuantizedTensor(key=k, group="other", bits=16, shape=tuple(w.shape),
                            dense=w if w.dtype == torch.bfloat16 else w.to(torch.bfloat16))
         for k, w in sd.items()}
    m = QuantRWKV7(QuantizedCheckpoint(tensors=t, config_repr="bf16", **meta))
    del t, sd
    for i in range(NSEQ):
        lg = m(mx.array(data[i:i + 1, :-1].numpy())).astype(mx.float32)
        mx.eval(lg)
        yield np.array(lg)[0]
        del lg


# ЭТАЛОН fp32 -- на CPU: на MPS fp32 у RWKV7Ref нет гарантии, что
# внутренние редукции не срежутся, а вопрос как раз о точности.
if not os.path.exists(REF32):
    t0 = time.time()
    V = 65536
    mm = np.lib.format.open_memmap(REF32, mode="w+", dtype=np.float32,
                                   shape=(NSEQ, SEQLEN - 1, V))
    for i, lg in enumerate(torch_logits(torch.float32, "cpu")):
        mm[i] = lg
        print("  эталон fp32 %d/%d" % (i + 1, NSEQ), flush=True)
    mm.flush(); del mm
    print("эталон fp32 за %.0f с" % (time.time() - t0), flush=True)

ref = np.load(REF32, mmap_mode="r")
res = {}
for name, gen in (("torch-bf16", lambda: torch_logits(torch.bfloat16, "mps")),
                  ("mlx-bf16path", mlx_logits)):
    ks, ts = [], []
    for i, got in enumerate(gen()):
        k, t = kl_stats(np.asarray(ref[i]), got)
        ks.append(float(k.mean())); ts.append(float(t.mean()))
    res[name] = (ks, ts)
    lo, hi = boot_ci(ks)
    print("%-14s KL(fp32‖·) = %.6f  CI [%.6f; %.6f]  top-1 %.3f%%"
          % (name, np.mean(ks), lo, hi, 100 * np.mean(ts)), flush=True)

d = np.array(res["torch-bf16"][0]) - np.array(res["mlx-bf16path"][0])
lo, hi = boot_ci(d)
print("\nэталон − MLX: %+.6f нат/ток  95%% CI [%+.6f; %+.6f]  %s"
      % (d.mean(), lo, hi, "ЗНАЧИМО" if lo * hi > 0 else "в шуме"))
print("положительная разность = НАШ ЭТАЛОН дальше от fp32, чем MLX-путь")
