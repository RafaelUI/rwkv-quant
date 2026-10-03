"""03.10, СЕРВЕР: записанный файл .rwkvq прибором статьи (RWKV7Ref, fp32-счёт, KL по окну к bf16-чекпоинту тем же
прибором) на ТРЁХ наборах: eval_text_heldout (36), eval_code_heldout (24, закрытый -- только для сверки с прежними
числами), eval_code_open (48, открытый -- в статью). Деквант читателя (fp16 / плотные bf16) кладётся в fp32 БЕЗ
округления (апкаст точен) -- как RWKVQ_FILE_FP32=1 у file_eval_srv_0210.
    python file_eval_0310.py <файл.rwkvq> <ckpt> <out.json>"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant.calibration import autopick as ap
from rwkv_quant.formats.reader import load_dequantized
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
F, CK, OUT = sys.argv[1:4]
DEV = os.environ.get("RWKVQ_DEVICE", "cuda"); DEVS = DEV.split(",") if "," in DEV else DEV
R = os.path.expanduser("~/rwkvq/")
t0 = time.time()
ev, cd, co = (torch.load(R + n, weights_only=False) for n in ("eval_text_heldout.pt", "eval_code_heldout.pt", "eval_code_open.pt"))
wins = ev["tokens"][:, :512].tolist() + cd["tokens"].tolist() + co["tokens"].tolist()
kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"]) + ["code_open"] * len(co["lang"])
langs = list(ev["lang"]) + list(cd["lang"]) + list(co["lang"])
assert len(wins) == len(kind) == 108 and all(len(w) == 512 for w in wins), (len(wins), len(kind))
sd = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)
dq = load_dequantized(F)
out, nq = {}, 0
for k, v in sd.items():
    w = dq[k]
    if tuple(w.shape) != tuple(v.shape):
        w = w.T if (w.dim() == 2 and tuple(w.T.shape) == tuple(v.shape)) else w.reshape(v.shape)
    nq += int(w.dtype == torch.float16)
    out[k] = w.float().contiguous()
    assert torch.isfinite(out[k]).all(), k
tmp = "/tmp/deq0310_%d_%s.pth" % (os.getpid(), os.path.basename(F))
torch.save(out, tmp); del out, dq, sd
print("деквант: fp16-тензоров %d, %.0f с" % (nq, time.time() - t0), flush=True)
W = torch.tensor(wins, dtype=torch.long)
BS = 11


def run(path, dt):
    M = RWKV7Ref(path, device=DEVS, dtype=dt, compute_dtype=torch.float32)
    data = W[:, :-1].contiguous().to(M.devices[0])
    I = ap._Instrument(M, data)
    with torch.no_grad():
        h = torch.cat([M.forward(data[i:i + BS], cfg=None, return_hidden=True).cpu() for i in range(0, data.shape[0], BS)])
    head, kdt, last = I.head32.cpu(), I.kdt, M.devices[-1]
    del M, I, data
    torch.cuda.empty_cache()
    return h, head, kdt, last


try:
    with torch.no_grad():
        href, head_ref, kdt, last = run(CK, torch.bfloat16)
        hq, head_q, _, _ = run(tmp, torch.float32)
        tgt = W[:, 1:].to(last); head_ref = head_ref.to(last); head_q = head_q.to(last)
        kl, ce, cer = [], [], []
        for j in range(hq.shape[0]):
            lp = torch.log_softmax((href[j].to(last) @ head_ref.T).to(kdt), -1)
            lq = torch.log_softmax((hq[j].to(last) @ head_q.T).to(kdt), -1)
            kl.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
            ce.append(float(-lq.gather(-1, tgt[j][:, None]).mean()))
            cer.append(float(-lp.gather(-1, tgt[j][:, None]).mean()))
finally:
    os.remove(tmp)
assert all(math.isfinite(x) and x >= 0 for x in kl) and all(math.isfinite(x) and x > 0 for x in ce + cer)
json.dump(dict(file=os.path.basename(F), bytes=os.path.getsize(F), ckpt=os.path.basename(CK), kind=kind, langs=langs,
               kl=kl, ce=ce, ce_ref=cer), open(OUT, "w"))
mean = lambda xs, k: sum(x for x, kk in zip(xs, kind) if kk == k) / kind.count(k)
for k in ("text", "code", "code_open"):
    print("%-9s KL %.6f | Δppl %+.3f%%" % (k, mean(kl, k), 100 * (math.exp(mean(ce, k) - mean(cer, k)) - 1)), flush=True)
print("%s: всего %.0f с" % (os.path.basename(F), time.time() - t0), flush=True)
