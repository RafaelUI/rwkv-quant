"""04.10: файл GGUF (llama-quantize из ТОГО ЖЕ pth) прибором статьи. Квантованные тензоры GGUF (Q4_K / Q6_K / ...)
деквантуются gguf-py и подставляются в state dict чекпоинта; прочее (F16 / F32 в GGUF) берётся из чекпоинта как есть
(bf16 -> f16 точно). Имена: output -> head, token_embd -> emb, blk.N.{time_mix_{receptance,key,value,output},
channel_mix_{key,value}} -> blocks.N.{att.*, ffn.*}; ориентация проверяется относительной ошибкой к исходному весу
(обязана быть < 0.5 ровно в одной из двух ориентаций либо формы совпадают однозначно).
Оценка -- как file_eval_0310 (text 36 / code 24 / code_open 48, KL к bf16-чекпоинту, fp32-счёт).
    python gguf_eval_0410.py <файл.gguf> <ckpt> <out.json> [--dry]   (--dry: только сопоставление и ошибки весов)"""
import json, math, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, gguf
from gguf import quants
F, CK, OUT = sys.argv[1:4]; DRY = "--dry" in sys.argv
t0 = time.time()
MAP = {"time_mix_receptance": "att.receptance", "time_mix_key": "att.key", "time_mix_value": "att.value", "time_mix_output": "att.output",
       "channel_mix_key": "ffn.key", "channel_mix_value": "ffn.value"}
def pth_key(name):
    if name == "output.weight": return "head.weight"
    if name == "token_embd.weight": return "emb.weight"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
    if m and m.group(2) in MAP: return "blocks.%s.%s.weight" % (m.group(1), MAP[m.group(2)])
    return None
sd = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)
r = gguf.GGUFReader(F)
out = {k: v.float().contiguous() for k, v in sd.items()}
types, errs, qbytes = {}, {}, 0
for t in r.tensors:
    tn = t.tensor_type.name
    types[tn] = types.get(tn, 0) + 1
    if tn in ("F16", "F32", "BF16"): continue
    k = pth_key(t.name)
    assert k is not None and k in sd, ("квантованный тензор без пары в чекпоинте", t.name, tn)
    w = torch.from_numpy(np.ascontiguousarray(quants.dequantize(t.data, t.tensor_type))).float()
    ref = out[k]
    cands = [c for c in (w, w.T) if tuple(c.shape) == tuple(ref.shape)]
    assert cands, (t.name, tuple(w.shape), tuple(ref.shape))
    e = [float((c - ref).norm() / ref.norm()) for c in cands]
    j = int(np.argmin(e))
    assert e[j] < 0.5 and (len(e) == 1 or e[1 - j] > 2 * e[j] or e[1 - j] > 0.5), (t.name, e)
    out[k] = cands[j].contiguous(); errs[k] = (tn, e[j]); qbytes += int(t.n_bytes)
    assert torch.isfinite(out[k]).all(), k
by = {}
for k, (tn, e) in errs.items():
    g = tn + " " + (k.split(".", 2)[2] if k.startswith("blocks.") else k)
    by.setdefault(g, []).append(e)
print("%s: тензоров %s; квантованных %d (%.1f МБ), файл %.1f МБ" % (os.path.basename(F), types, len(errs), qbytes / 1e6, os.path.getsize(F) / 1e6), flush=True)
for g in sorted(by): print("   %-28s n=%3d  ||dW||/||W|| среднее %.4f, макс %.4f" % (g, len(by[g]), sum(by[g]) / len(by[g]), max(by[g])), flush=True)
if DRY: sys.exit(0)
from rwkv_quant.calibration import autopick as ap
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
DEV = os.environ.get("RWKVQ_DEVICE", "cuda"); DEVS = DEV.split(",") if "," in DEV else DEV
R = os.path.expanduser("~/rwkvq/")
ev, cd, co = (torch.load(R + n, weights_only=False) for n in ("eval_text_heldout.pt", "eval_code_heldout.pt", "eval_code_open.pt"))
wins = ev["tokens"][:, :512].tolist() + cd["tokens"].tolist() + co["tokens"].tolist()
kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"]) + ["code_open"] * len(co["lang"])
langs = list(ev["lang"]) + list(cd["lang"]) + list(co["lang"])
assert len(wins) == 108
tmp = "/tmp/gguf0410_%d_%s.pth" % (os.getpid(), os.path.basename(F))
torch.save(out, tmp); del out, sd
W = torch.tensor(wins, dtype=torch.long); BS = 11


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
assert all(math.isfinite(x) and x >= 0 for x in kl)
json.dump(dict(file=os.path.basename(F), bytes=os.path.getsize(F), ckpt=os.path.basename(CK), kind=kind, langs=langs, kl=kl, ce=ce, ce_ref=cer,
               types=types, werr={k: v for k, v in errs.items()}), open(OUT, "w"))
mean = lambda xs, k: sum(x for x, kk in zip(xs, kind) if kk == k) / kind.count(k)
for k in ("text", "code", "code_open"):
    print("%-9s KL %.6f | Δppl %+.3f%%" % (k, mean(kl, k), 100 * (math.exp(mean(ce, k) - mean(cer, k)) - 1)), flush=True)
print("всего %.0f с" % (time.time() - t0), flush=True)
