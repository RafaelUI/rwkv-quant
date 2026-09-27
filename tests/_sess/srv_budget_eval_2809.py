"""28.09, СЕРВЕР: копия srv_budget_eval_2409 с ЧИСТЫМ текстом -- eval_text_heldout.pt (Википедия
en/ru/sr по 12 окон, ни одной 16-граммы с calib_corpus) вместо eval_corpus_multiling (он целиком
внутри калибровочного корпуса, 27.09). Перед замером -- проверка пересечения 16-грамм eval-окон с
calib_corpus: текст обязан дать 0 (иначе отказ), код печатается (1 из 24 окон, 0.03%, известно).
Остальное как в 24.09:
24.09, СЕРВЕР: как budget_eval_all_2409, но устройство(а) из RWKVQ_DEVICE ("cuda:1,cuda:2" --
слои по картам), токенизатор и пути -- аргументами, выборы -- select_budget прямо из measure JSON.
    python srv_budget_eval_2409.py <ckpt> <tok> <measure.json> <out.json> <бюджеты через запятую>

24.09: вариант budget_eval_2409 -- окна: 38 eval_corpus_multiling (ru/en/sr) + 24 отложенного
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
CK, TOK, MJ, OUT = sys.argv[1:5]
BUDGETS = [float(b) for b in sys.argv[5].split(",")]
DEV = os.environ.get("RWKVQ_DEVICE", "cuda")
DEVS = DEV.split(",") if "," in DEV else DEV
MEAS = json.load(open(MJ))
SELS = []
for b in BUDGETS:
    o, r = ap.select_budget(MEAS, b)
    SELS.append(("wprop_%+.3f" % b, o))
    print("бюджет %+.3f: tau %.3f, %+.3f%% файла, вверх %d, вниз %d, предск. KL %+.1f%%" % (b, r["tau"], 100 * r["bytes_frac"], r["n_up"], r["n_down"], 100 * r["kl_pred"]), flush=True)
sw = lambda: float(open("/proc/meminfo").read().split("SwapFree:")[0].split("SwapTotal:")[1].split()[0]) / 1024 * 0  # своп сервера не меряем
s0 = sw(); t0 = time.time()
ev = torch.load(os.path.expanduser(os.environ.get("RWKVQ_EVAL_TEXT", "~/rwkvq/eval_text_heldout.pt")))
print("текст:", os.environ.get("RWKVQ_EVAL_TEXT", "~/rwkvq/eval_text_heldout.pt"), len(ev["lang"]), "окон", flush=True)
wins = [r for r in ev["tokens"][:, :512].tolist()]; langs = list(ev["lang"]); kind = ["eval"] * len(wins)
cd = torch.load(os.path.expanduser("~/rwkvq/eval_code_heldout.pt"))
wins += cd["tokens"].tolist(); langs += list(cd["lang"]); kind += ["code"] * len(cd["lang"])
import re as _re
_enc = A._encoder(TOK)
_G = set()
for _c in _re.split(r"—+ CHUNK —+", open(A.CORPUS, encoding="utf-8").read()):
    _ids = _enc(_c.strip()); _G.update(tuple(_ids[i:i + 16]) for i in range(len(_ids) - 16))
_ov = [sum(tuple(w[i:i + 16]) in _G for i in range(len(w) - 16)) / (len(w) - 16) for w in wins]
_nt = sum(1 for j, o in enumerate(_ov) if o > 0 and kind[j] == "eval"); _nc = sum(1 for j, o in enumerate(_ov) if o > 0 and kind[j] == "code")
print("пересечение 16-грамм с calib_corpus: текст %d/%d окон, код %d/%d (макс. доля %.4f)" % (_nt, kind.count("eval"), _nc, kind.count("code"), max(_ov)), flush=True)
assert _nt == 0, "отложенный текст пересекается с калибровкой"
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
M = RWKV7Ref(CK, device=DEVS, dtype=torch.bfloat16, compute_dtype=torch.float32)
W = torch.tensor(wins, dtype=torch.long)
data = W[:, :-1].contiguous().to(M.devices[0]); tgt = W[:, 1:].contiguous().to(M.devices[-1])
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
            c.bits_overrides = dict(sel, **cfg.bits_overrides)
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
