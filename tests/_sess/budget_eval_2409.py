"""Цель правила при РАВНЫХ БАЙТАХ (24.09): выборы select_budget по двум измерениям g1j 1.5B
(квота en3/code2/ru1/sr1/zh1 и квота en3/ru3/sr2) при бюджетах 0 и +0.5% файла.
Окна: 38 окон eval_corpus_multiling (ru20/en9/sr9, ни одно не в выборе) + 6 окон кода/zh
калибровочного корпуса (code4/zh2; в выборе en/code из них code 2 и zh 1, в ru/en/sr -- ни одного).
Прибор -- как autopick.measure (RWKV7Ref bf16-хранение, fp32-счёт, эталон тем же прибором).
KL и CE по окну; Δppl по eval-окнам.
    python budget_eval_2409.py <ckpt> <out.json> имя=выбор.json ..."""
import copy, json, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK, OUT = sys.argv[1], sys.argv[2]
SELS = [a.split("=", 1) for a in sys.argv[3:]]
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split("M")[0])
s0 = sw(); t0 = time.time()
ev = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
wins = [r for r in ev["tokens"][:, :512].tolist()]; langs = list(ev["lang"]); kind = ["eval"] * len(wins)
cw, cl = ap._pick_windows(A._encoder(TOK), A.CORPUS, ap.SEQ_LEN, (("code", 4), ("zh", 2)))
wins += cw; langs += cl; kind += ["calib"] * len(cw)
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
M = RWKV7Ref(CK, device="mps", dtype=torch.bfloat16, compute_dtype=torch.float32)
W = torch.tensor(wins, dtype=torch.long)
data = W[:, :-1].contiguous().to("mps"); tgt = W[:, 1:].contiguous().to("mps")
I = ap._Instrument(M, data)
BS = 11


def hidden(c):
    return torch.cat([M.forward(data[i:i + BS], cfg=c, return_hidden=True) for i in range(0, data.shape[0], BS)])


def stats(h, head):
    kl, ce = [], []
    for j in range(h.shape[0]):
        lp = torch.log_softmax((href[j] @ I.head32.T).to(I.kdt), -1)
        lq = torch.log_softmax((h[j] @ head.T).to(I.kdt), -1)
        kl.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        ce.append(float(-lq.gather(-1, tgt[j][:, None]).mean()))
    return kl, ce


res = dict(ckpt=os.path.basename(CK), langs=langs, kind=kind, kl={}, ce={})
with torch.no_grad():
    href = hidden(None)
    res["ce"]["ref"] = stats(href, I.head32)[1]
    print("эталон %.0f с" % (time.time() - t0), flush=True)
    for name, sel in [("live", None)] + SELS:
        c = copy.copy(cfg)
        if sel:
            c.bits_overrides = dict(json.load(open(sel)), **cfg.bits_overrides)
        t1 = time.time()
        res["kl"][name], res["ce"][name] = stats(hidden(c), M._q(M.head_weight, "head", c, "head.weight"))
        json.dump(res, open(OUT, "w"), indent=1)
        print("%-10s %.0f с" % (name, time.time() - t1), flush=True)
import math
ev_i = [j for j in range(len(langs)) if kind[j] == "eval"]
def ppl(n):
    return math.exp(sum(res["ce"][n][j] for j in ev_i) / len(ev_i))
L = res["kl"]["live"]
print("%-10s %8s %8s %8s %8s %8s %8s %9s" % ("", "eval KL", "en", "ru", "sr", "code*", "zh*", "Δppl eval"))
for n in res["kl"]:
    K = res["kl"][n]
    def g(pred):
        idx = [j for j in range(len(langs)) if pred(j)]
        return sum(K[j] for j in idx) / len(idx), sum(L[j] for j in idx) / len(idx)
    row = [g(lambda j: kind[j] == "eval")] + [g(lambda j, l=l: kind[j] == "eval" and langs[j] == l) for l in ("en", "ru", "sr")] + \
          [g(lambda j, l=l: kind[j] == "calib" and langs[j] == l) for l in ("code", "zh")]
    print("%-10s " % n + " ".join("%.5f" % a if n == "live" else "%+7.1f%%" % (100 * (a / b - 1)) for a, b in row) +
          "  %+.3f%%" % (100 * (ppl(n) / ppl("ref") - 1)), flush=True)
print("* code/zh -- окна КАЛИБРОВОЧНОГО корпуса; всего %.0f с, своп %+.0f МБ" % (time.time() - t0, sw() - s0), flush=True)
