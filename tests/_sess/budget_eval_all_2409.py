"""24.09: вариант budget_eval_2409 -- окна: 38 eval_corpus_multiling (ru/en/sr) + 24 отложенного
кода (eval_code_heldout.pt, py/swift); ни одно не в выборе. Сводка по языкам и Δppl по обоим наборам.

Цель правила при РАВНЫХ БАЙТАХ (24.09): выборы select_budget по двум измерениям g1j 1.5B
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
ev = torch.load(os.path.expanduser(os.environ.get("RWKVQ_EVAL_TEXT", "~/Develop/WKV-kvant/eval_corpus_multiling.pt")))  # 27.09: Википедия -- eval_text_heldout.pt
wins = [r for r in ev["tokens"][:, :512].tolist()]; langs = list(ev["lang"]); kind = ["eval"] * len(wins)
cd = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_code_heldout.pt"))
wins += cd["tokens"].tolist(); langs += list(cd["lang"]); kind += ["code"] * len(cd["lang"])
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
def ppl(n, k):
    ix = [j for j in range(len(langs)) if kind[j] == k]
    return math.exp(sum(res["ce"][n][j] for j in ix) / len(ix))
L = res["kl"]["live"]
cols = [("текст", lambda j: kind[j] == "eval"), ("en", lambda j: langs[j] == "en"), ("ru", lambda j: langs[j] == "ru"),
        ("sr", lambda j: langs[j] == "sr"), ("код", lambda j: kind[j] == "code"), ("py", lambda j: langs[j] == "py"),
        ("swift", lambda j: langs[j] == "swift")]
print("%-12s" % "" + "".join("%9s" % c for c, _ in cols) + "  Δppl текст  Δppl код")
for n in res["kl"]:
    K = res["kl"][n]
    cells = []
    for _, pr in cols:
        ix = [j for j in range(len(langs)) if pr(j)]
        a_, b_ = sum(K[j] for j in ix) / len(ix), sum(L[j] for j in ix) / len(ix)
        cells.append("%9.5f" % a_ if n == "live" else "%+8.1f%%" % (100 * (a_ / b_ - 1)))
    print("%-12s" % n + "".join(cells) + "   %+.3f%%    %+.3f%%" % (100 * (ppl(n, "eval") / ppl("ref", "eval") - 1),
                                                                    100 * (ppl(n, "code") / ppl("ref", "code") - 1)), flush=True)
print("всего %.0f с, своп %+.0f МБ" % (time.time() - t0, sw() - s0), flush=True)
