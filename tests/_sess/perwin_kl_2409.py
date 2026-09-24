"""Цель правила (п. 2, 24.09): KL пресета ПО КАЖДОМУ калибровочному окну и что с ним делает
выбор tau5 при двух квотах. Прибор -- тот же, что autopick.measure (RWKV7Ref bf16-хранение,
fp32-счёт, эталон тем же прибором, KL окно за окном).
Окна: квота en6/code4/ru3/sr2/zh2 -- её префиксы по языку совпадают с окнами обеих квот
измерения (en3/code2/ru1/sr1/zh1 и en3/ru3/sr2), см. _pick_windows (порядок внутри языка
не зависит от n).
    python perwin_kl_2409.py <ckpt> <sel_encode.json> <sel_rues.json> <out.json>"""
import copy, json, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
CK, SEL_A, SEL_B, OUT = sys.argv[1:5]
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split("M")[0])
s0 = sw(); t0 = time.time()
Q = (("en", 6), ("code", 4), ("ru", 3), ("sr", 2), ("zh", 2))
wins, langs = ap._pick_windows(A._encoder(TOK), A.CORPUS, ap.SEQ_LEN, Q)
inA = [];  inB = []
cnt = {}
for l in langs:
    cnt[l] = cnt.get(l, 0) + 1
    inA.append(cnt[l] <= dict(ap.QUOTA).get(l, 0))
    inB.append(cnt[l] <= {"en": 3, "ru": 3, "sr": 2}.get(l, 0))
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
M = RWKV7Ref(CK, device="mps", dtype=torch.bfloat16, compute_dtype=torch.float32)
data = torch.tensor(wins, dtype=torch.long)[:, :-1].contiguous().to("mps")
I = ap._Instrument(M, data)


def perwin(h, head):
    out = []
    for j in range(h.shape[0]):
        lp = torch.log_softmax((href[j] @ I.head32.T).to(I.kdt), -1)
        lq = torch.log_softmax((h[j] @ head.T).to(I.kdt), -1)
        out.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
    return out


res = dict(ckpt=os.path.basename(CK), langs=langs, inA=inA, inB=inB, kl={})
with torch.no_grad():
    href = M.forward(data, return_hidden=True)
    for name, sel in (("live", None), ("tau5_encode", SEL_A), ("tau5_rues", SEL_B)):
        c = copy.copy(cfg)
        if sel:
            c.bits_overrides = dict(json.load(open(sel)), **cfg.bits_overrides)
        t1 = time.time()
        h = M.forward(data, cfg=c, return_hidden=True)
        res["kl"][name] = perwin(h, M._q(M.head_weight, "head", c, "head.weight"))
        print("%-12s %.0f с  среднее %.5f" % (name, time.time() - t1, sum(res["kl"][name]) / len(langs)), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
L = res["kl"]["live"]
print("окно  язык  A B   live      tau5_encode       tau5_rues")
for j, l in enumerate(langs):
    a, b = res["kl"]["tau5_encode"][j], res["kl"]["tau5_rues"][j]
    print("%3d  %-4s  %s %s  %.5f  %.5f (%+5.1f%%)  %.5f (%+5.1f%%)" % (
        j, l, "A" if inA[j] else ".", "B" if inB[j] else ".", L[j], a, 100 * (a / L[j] - 1), b, 100 * (b / L[j] - 1)))
for grp in ("A", "B"):
    idx = [j for j in range(len(langs)) if (inA if grp == "A" else inB)[j]]
    tot = sum(L[j] for j in idx)
    print("квота %s: доля KL live по языкам: %s" % (grp, {l: round(sum(L[j] for j in idx if langs[j] == l) / tot, 3) for l in dict.fromkeys(langs[j] for j in idx)}))
print("всего %.0f с, своп %+.0f МБ" % (time.time() - t0, sw() - s0), flush=True)
