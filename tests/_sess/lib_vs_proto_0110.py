"""01.10, СЕРВЕР: библиотечный GPTQ (calibration.gptq.run, тот, что зовёт quantize()) против прототипа, по которому
мерились таблицы (gptq_srv_2809 / gptq_srv_3009). Тот же прибор, что matrix_eval_3009 (36 окон текста + 24 кода, KL
к bf16), та же калибровка (600 окон v2), damp 0.1. Совпадение KL/CE с JSON прототипа -- доказательство, что умолчание
quantize() даёт ИЗМЕРЕННОЕ качество, а не только файл == fake-путь библиотеки.
    python lib_vs_proto_0110.py <ckpt> <reduction|compression> <JSON прототипа>"""
import copy, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap, gptq as G
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK, PRESET, PJ = sys.argv[1], sys.argv[2], sys.argv[3]
TOK = os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
DEV = os.environ.get("RWKVQ_DEVICE", "cuda")
t0 = time.time()
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.REDUCTION if PRESET == "reduction" else presets.COMPRESSION)
cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
cal = torch.load(os.path.expanduser("~/rwkvq/gptq_calib2_2809.pt"), weights_only=False)["tokens"][:600]
qts, deqs = G.run(CK, TOK, cfg, n_windows=600, damp=0.1, device=DEV, keep_deq=True, verbose=False, calib=cal)
print("GPTQ библиотеки: %d матриц, %.0f с" % (len(qts), time.time() - t0), flush=True)
ev = torch.load(os.path.expanduser("~/rwkvq/eval_text_heldout.pt")); cd = torch.load(os.path.expanduser("~/rwkvq/eval_code_heldout.pt"))
wins = ev["tokens"][:, :512].tolist() + cd["tokens"].tolist()
kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"])
M = RWKV7Ref(CK, device=DEV, dtype=torch.bfloat16, compute_dtype=torch.float32)
W = torch.tensor(wins, dtype=torch.long)
data = W[:, :-1].contiguous().to(M.devices[0]); tgt = W[:, 1:].contiguous().to(M.devices[-1])
I = ap._Instrument(M, data)
BS = 11
hidden = lambda c: torch.cat([M.forward(data[i:i + BS], cfg=c, return_hidden=True) for i in range(0, data.shape[0], BS)])
with torch.no_grad():
    href = hidden(None)
    for key, Q in deqs.items():
        if key == "head.weight":
            M.head_weight = Q.to(M.head_weight.device); continue
        i, rest = int(key.split(".")[1]), key.split(".", 2)[2]
        obj = M.tmix[i] if rest.startswith("att.") else M.cmix[i]
        attr = {"att.receptance.weight": "r_proj", "att.key.weight": "k_proj", "att.value.weight": "v_proj",
                "att.output.weight": "o_proj", "ffn.key.weight": "key", "ffn.value.weight": "value"}[rest]
        setattr(obj, attr, Q.to(getattr(obj, attr).device))
    c = copy.copy(cfg)
    c.bits_overrides = dict({k: 16 for k in deqs}, **{k: v for k, v in cfg.bits_overrides.items() if k not in deqs})
    h = hidden(c)
    head = M._q(M.head_weight, "head", c, "head.weight")
    kl, ce = [], []
    for j in range(h.shape[0]):
        lp = torch.log_softmax((href[j] @ I.head32.T).to(I.kdt), -1)
        lq = torch.log_softmax((h[j] @ head.T).to(I.kdt), -1)
        kl.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        ce.append(float(-lq.gather(-1, tgt[j][:, None]).mean()))
p = json.load(open(PJ))
mean = lambda xs, k: sum(x for x, kk in zip(xs, kind) if kk == k) / kind.count(k)
for k in ("text", "code"):
    a, b = mean(kl, k), mean(p["kl"]["gptq"], k)
    ca, cb = mean(ce, k), mean(p["ce"]["gptq"], k)
    print("%-4s KL библиотека %.6f, прототип %.6f (%+.2f%%) | CE %.6f / %.6f (Δppl %+.4f%%)" % (k, a, b, 100 * (a / b - 1), ca, cb, 100 * (math.exp(ca - cb) - 1)), flush=True)
print("всего %.0f с" % (time.time() - t0))
