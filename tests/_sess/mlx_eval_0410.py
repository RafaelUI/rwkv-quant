"""04.10, СЕРВЕР: оценка схемы MLX (коды mx.quantize с Mac, mlx_export_0410.py) прибором статьи -- как file_eval_0310
(text 36 / code 24 / code_open 48, KL к bf16-чекпоинту, fp32-счёт). Деквант: q * scale + bias в fp32 -> fp16 (побитно
mx.dequantize; сверяется контрольной суммой из .npz).
    python mlx_eval_0410.py <файл.npz> <ckpt> <out.json>"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from rwkv_quant.calibration import autopick as ap
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
F, CK, OUT = sys.argv[1:4]
DEV = os.environ.get("RWKVQ_DEVICE", "cuda"); DEVS = DEV.split(",") if "," in DEV else DEV
R = os.path.expanduser("~/rwkvq/"); t0 = time.time()
ev, cd, co = (torch.load(R + n, weights_only=False) for n in ("eval_text_heldout.pt", "eval_code_heldout.pt", "eval_code_open.pt"))
wins = ev["tokens"][:, :512].tolist() + cd["tokens"].tolist() + co["tokens"].tolist()
kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"]) + ["code_open"] * len(co["lang"])
langs = list(ev["lang"]) + list(cd["lang"]) + list(co["lang"])
assert len(wins) == 108
W = torch.tensor(wins, dtype=torch.long); BS = 11
Z = np.load(F)
info = Z["__info__"]


def deq(name):
    bits, o, i, chk = (int(x) for x in Z[name + "|meta"])
    wq, s, b = Z[name + "|wq"], Z[name + "|s"], Z[name + "|b"]
    bitsarr = np.unpackbits(np.ascontiguousarray(wq).view(np.uint8).reshape(o, -1), axis=1, bitorder="little")[:, : i * bits].reshape(o, i, bits)
    q = (bitsarr << np.arange(bits, dtype=np.uint8)).sum(-1, dtype=np.uint8)
    qt = torch.from_numpy(q).reshape(o, i // 64, 64).float()
    d = (qt * torch.from_numpy(s).float()[..., None] + torch.from_numpy(b).float()[..., None]).to(torch.float16).reshape(o, i)
    got = int(np.frombuffer(d.numpy().tobytes(), dtype=np.uint16).astype(np.uint64).sum() % (1 << 62))
    assert got == chk, ("деквант не равен mx.dequantize", name, got, chk)
    return d.float()


def hidden(M):
    data = W[:, :-1].contiguous().to(M.devices[0])
    I = ap._Instrument(M, data)
    with torch.no_grad():
        h = torch.cat([M.forward(data[i:i + BS], cfg=None, return_hidden=True).cpu() for i in range(0, data.shape[0], BS)])
    return h, I.kdt


with torch.no_grad():
    M = RWKV7Ref(CK, device=DEVS, dtype=torch.bfloat16, compute_dtype=torch.float32)
    href, kdt = hidden(M); head_ref = M.head_weight.float().cpu(); last = M.devices[-1]
    del M; torch.cuda.empty_cache()
    M = RWKV7Ref(CK, device=DEVS, dtype=torch.float32, compute_dtype=torch.float32)
    names = sorted(k[:-5] for k in Z.files if k.endswith("|meta")); worst = 0.0
    for name in names:
        p = name.split(".")
        obj = M if len(p) == 1 else getattr(M, p[0])[int(p[1])]
        attr = p[-1]; w = getattr(obj, attr); d = deq(name)
        assert tuple(d.shape) == tuple(w.shape), (name, tuple(d.shape), tuple(w.shape))
        e = float((d.to(w.device) - w.float()).norm() / w.float().norm()); worst = max(worst, e)
        assert e < 0.6, (name, e)
        setattr(obj, attr, d.to(w.device))
    print("подставлено %d тензоров, макс ||dW||/||W|| %.4f, %.0f с" % (len(names), worst, time.time() - t0), flush=True)
    hq, _ = hidden(M); head_q = M.head_weight.float().cpu()
    del M; torch.cuda.empty_cache()
    tgt = W[:, 1:].to(last); head_ref = head_ref.to(last); head_q = head_q.to(last)
    kl, ce, cer = [], [], []
    for j in range(hq.shape[0]):
        lp = torch.log_softmax((href[j].to(last) @ head_ref.T).to(kdt), -1)
        lq = torch.log_softmax((hq[j].to(last) @ head_q.T).to(kdt), -1)
        kl.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        ce.append(float(-lq.gather(-1, tgt[j][:, None]).mean()))
        cer.append(float(-lp.gather(-1, tgt[j][:, None]).mean()))
assert all(math.isfinite(x) and x >= 0 for x in kl)
json.dump(dict(file=os.path.basename(F), bytes=float(info[2]), bits_main=int(info[0]), ckpt=os.path.basename(CK), kind=kind, langs=langs,
               kl=kl, ce=ce, ce_ref=cer), open(OUT, "w"))
mean = lambda xs, k: sum(x for x, kk in zip(xs, kind) if kk == k) / kind.count(k)
for k in ("text", "code", "code_open"):
    print("%-9s KL %.6f | Δppl %+.3f%%" % (k, mean(kl, k), 100 * (math.exp(mean(ce, k) - mean(cer, k)) - 1)), flush=True)
print("оценка размера %.1f МБ; всего %.0f с" % (info[2] / 1e6, time.time() - t0), flush=True)
