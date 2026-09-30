"""30.09, СЕРВЕР: сводная таблица деградации по размеру модели и пресету -- ТОТ ЖЕ прибор, что srv_budget_eval_2809 и
gptq_srv_2809 (RWKV7Ref bf16-хранение, fp32-счёт; eval_text_heldout 36 окон + eval_code_heldout 24; KL по окну к bf16
тем же прибором; CE -> Δppl). Плечи (fake-путь; файл == fake-путь побитно -- гейты):
  red   -- REDUCTION (RTN, act_stats как у quantize)
  comp  -- COMPRESSION без autopick (live)
  wprop -- COMPRESSION + autopick +0.5% (measure: путь или auto -- ap.measure на этом устройстве, копия в ~/rwkvq/measure_<метка>.json)
    python matrix_eval_3009.py <ckpt> <метка> <плечи через запятую> [measure.json|auto]   -> ~/rwkvq/matrix_<метка>.json"""
import copy, json, math, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK, LAB, ARMS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
MJ = sys.argv[4] if len(sys.argv) > 4 else None
TOK = os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
OUT = os.path.expanduser("~/rwkvq/matrix_%s.json" % LAB)
DEV = os.environ.get("RWKVQ_DEVICE", "cuda")
DEVS = DEV.split(",") if "," in DEV else DEV
t0 = time.time()
print("старт %s %s плечи %s measure %s DEV %s" % (time.ctime(), os.path.basename(CK), ARMS, MJ, DEV), flush=True)
_, sig = A.collect(CK, TOK)
cfgs = {}
for name, base in (("red", presets.REDUCTION), ("comp", presets.COMPRESSION)):
    c = copy.deepcopy(base); c.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig); cfgs[name] = c
if "wprop" in ARMS:
    if MJ == "auto":
        m = ap.measure(CK, cfgs["comp"], TOK, device=DEV)
        MJ = os.path.expanduser("~/rwkvq/measure_%s.json" % LAB)
        json.dump(m, open(MJ, "w"))
        print("measure: %.0f с -> %s" % (m.get("seconds", -1), MJ), flush=True)
    m = json.load(open(MJ))
    o, r = ap.select_budget(m, 0.005)
    c = copy.copy(cfgs["comp"]); c.bits_overrides = dict(o, **cfgs["comp"].bits_overrides); cfgs["wprop"] = c
    print("autopick +0.5%%: tau %.3f, %+.3f%% файла, вверх %d, вниз %d" % (r["tau"], 100 * r["bytes_frac"], r["n_up"], r["n_down"]), flush=True)
ev = torch.load(os.path.expanduser("~/rwkvq/eval_text_heldout.pt"))
cd = torch.load(os.path.expanduser("~/rwkvq/eval_code_heldout.pt"))
wins = ev["tokens"][:, :512].tolist() + cd["tokens"].tolist()
langs = list(ev["lang"]) + list(cd["lang"]); kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"])
M = RWKV7Ref(CK, device=DEVS, dtype=torch.bfloat16, compute_dtype=torch.float32)
W = torch.tensor(wins, dtype=torch.long)
data = W[:, :-1].contiguous().to(M.devices[0]); tgt = W[:, 1:].contiguous().to(M.devices[-1])
I = ap._Instrument(M, data)
BS = 11
hidden = lambda c: torch.cat([M.forward(data[i:i + BS], cfg=c, return_hidden=True) for i in range(0, data.shape[0], BS)])


def stats(h, head):
    kl, ce = [], []
    for j in range(h.shape[0]):
        lp = torch.log_softmax((href[j] @ I.head32.T).to(I.kdt), -1)
        lq = torch.log_softmax((h[j] @ head.T).to(I.kdt), -1)
        kl.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        ce.append(float(-lq.gather(-1, tgt[j][:, None]).mean()))
    return kl, ce


res = dict(ckpt=os.path.basename(CK), langs=langs, kind=kind, kl={}, ce={}, measure=MJ)
with torch.no_grad():
    href = hidden(None)
    res["ce"]["ref"] = stats(href, I.head32)[1]
    for a in ARMS:
        t1 = time.time()
        res["kl"][a], res["ce"][a] = stats(hidden(cfgs[a]), M._q(M.head_weight, "head", cfgs[a], "head.weight"))
        json.dump(res, open(OUT, "w"))
        print("%-6s %.0f с" % (a, time.time() - t1), flush=True)
mean = lambda xs, k: sum(x for x, kk in zip(xs, kind) if kk == k) / kind.count(k)
ppl = lambda n, k: math.exp(mean(res["ce"][n], k))
for a in ARMS:
    print("%-6s KL текст %.5f код %.5f | Δppl текст %+.3f%% код %+.3f%%" % (a, mean(res["kl"][a], "text"), mean(res["kl"][a], "code"),
          100 * (ppl(a, "text") / ppl("ref", "text") - 1), 100 * (ppl(a, "code") / ppl("ref", "code") - 1)), flush=True)
print("всего %.0f с" % (time.time() - t0), flush=True)
