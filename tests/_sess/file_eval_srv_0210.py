"""02.10, СЕРВЕР: ЗАПИСАННЫЙ файл .rwkvq тем же прибором, что matrix_eval_3009 / gptq_srv (36+24 окна, KL к bf16, CE):
деквант файла (reader.load_dequantized) подставляется в state dict плотного чекпоинта, RWKV7Ref считает без fake-квантования.
Сравнение с JSON прототипа (плечо gptq) -- отделяет «файл не тот» от «прибор не тот».
    python file_eval_srv_0210.py <файл.rwkvq> <ckpt> <JSON прототипа>"""
import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json, torch
from rwkv_quant.calibration import autopick as ap
from rwkv_quant.formats.reader import load_dequantized
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
F, CK, PJ = sys.argv[1:4]
FP32 = os.environ.get("RWKVQ_FILE_FP32") == "1"
if FP32:
    # 02.10: деквант БЕЗ финального округления в bf16 (reader округляет; fake-путь и прототип GPTQ держат fp32) --
    # тексты функций reader берутся как есть, вычёркивается только .to(torch.bfloat16)
    import inspect
    from rwkv_quant.formats import reader as _rd
    for _fn in ("_dequantize_gw_sb6", "_dequantize_gw_sym"):
        _src = inspect.getsource(getattr(_rd, _fn)).replace(".to(torch.bfloat16)", "")
        exec(compile(_src, _rd.__file__, "exec"), _rd.__dict__)
    _probe = inspect.getsource(_rd._dequantize_one)
t0 = time.time()
sd = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)
dq = load_dequantized(F)
out = {}
for k, v in sd.items():
    w = dq[k]
    if tuple(w.shape) != tuple(v.shape):
        w = w.T if (w.dim() == 2 and tuple(w.T.shape) == tuple(v.shape)) else w.reshape(v.shape)
    out[k] = (w.float() if FP32 else w.to(v.dtype)).contiguous()
tmp = "/tmp/deq_%s.pth" % os.path.basename(F)
torch.save(out, tmp); del out, dq
ev = torch.load(os.path.expanduser("~/rwkvq/eval_text_heldout.pt")); cd = torch.load(os.path.expanduser("~/rwkvq/eval_code_heldout.pt"))
wins = ev["tokens"][:, :512].tolist() + cd["tokens"].tolist()
kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"])
DEV = os.environ.get("RWKVQ_DEVICE", "cuda")
W = torch.tensor(wins, dtype=torch.long)
BS = 11


def run(path, dt=torch.bfloat16):
    M = RWKV7Ref(path, device=DEV, dtype=dt, compute_dtype=torch.float32)
    data = W[:, :-1].contiguous().to(M.devices[0])
    I = ap._Instrument(M, data)
    with torch.no_grad():
        h = torch.cat([M.forward(data[i:i + BS], cfg=None, return_hidden=True) for i in range(0, data.shape[0], BS)])
    return h, I.head32, I.kdt


with torch.no_grad():
    href, head_ref, kdt = run(CK)
    hq, head_q, _ = run(tmp, torch.float32 if FP32 else torch.bfloat16)
    tgt = W[:, 1:].to(hq.device)
    kl, ce = [], []
    for j in range(hq.shape[0]):
        lp = torch.log_softmax((href[j] @ head_ref.T).to(kdt), -1)
        lq = torch.log_softmax((hq[j] @ head_q.T).to(kdt), -1)
        kl.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        ce.append(float(-lq.gather(-1, tgt[j][:, None]).mean()))
os.remove(tmp)
p = json.load(open(PJ))
mean = lambda xs, k: sum(x for x, kk in zip(xs, kind) if kk == k) / kind.count(k)
for k in ("text", "code"):
    print("%-4s KL файл %.6f, прототип %.6f (%+.2f%%) | Δppl к эталону: файл %+.3f%%, прототип %+.3f%%" % (
        k, mean(kl, k), mean(p["kl"]["gptq"], k), 100 * (mean(kl, k) / mean(p["kl"]["gptq"], k) - 1),
        100 * (math.exp(mean(ce, k) - mean(p["ce"]["ref"], k)) - 1), 100 * (math.exp(mean(p["ce"]["gptq"], k) - mean(p["ce"]["ref"], k)) - 1)), flush=True)
json.dump(dict(kl=kl, ce=ce, kind=kind), open(os.path.expanduser("~/rwkvq/file_eval_%s%s.json" % (os.path.basename(F), "_fp32" if FP32 else "")), "w"))
print("всего %.0f с" % (time.time() - t0))
