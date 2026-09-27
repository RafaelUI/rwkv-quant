"""GPTQ (компенсация ошибки округления) для COMPRESSION, прототип (27.09).

Сетка -- ТА ЖЕ, что у asym_sb6_aw (блок 32, суперблок 8, 6-битные scale/min, поиск scale
с весами E[x^2]): scale/min суперблока (256 колонок) считаются В НАЧАЛЕ суперблока по уже
поправленным весам, затем колонки квантуются по одной на эту сетку, а ошибка каждой колонки
разносится на ещё не квантованные через H^-1 (Frantar et al. 2022). Формат файла не меняется:
те же коды/scale/min. Последовательно по слоям: вход каждой матрицы снимается с модели, где
предыдущие матрицы уже квантованы (r,k,v -> o -> ffn.key -> ffn.value; голова -- в конце).
Оценка в том же процессе: live (RTN-AW, как сейчас) против GPTQ на 38 окнах текста + 24 кода
(ни одно не калибровочное), прибор как в budget_eval (RWKV7Ref bf16-хранение, fp32-счёт).
    python gptq_proto_2709.py <ckpt> <out.json> [окон калибровки=48] [damp=0.01] [autopick.json]
Проверка 0: при диагональной H GPTQ обязан ПОБИТНО совпасть с RTN (_gw_one) -- иначе сетка не та."""
import copy, json, math, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
import torch.nn.functional as F
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap, groupwise as gw
from rwkv_quant.models import rwkv7_ref as R
CK, OUT = sys.argv[1], sys.argv[2]
NCAL = int(sys.argv[3]) if len(sys.argv) > 3 else 48
DAMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
OVR = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else {}
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
DEV = "mps"
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split("M")[0])
s0 = sw(); t0 = time.time()
GS, SB = 32, 8


def gptq(W, H, bits, ex2, damp=DAMP):
    """W [OUT, IN] fp32 (на DEV), H [IN, IN] fp32 (на CPU) -> деквантованные веса [OUT, IN] fp32."""
    OUTd, IN = W.shape
    assert IN % (GS * SB) == 0, IN
    W = W.clone(); H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead.to(W.device)] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(IN, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float().to(W.device)
    Q = torch.zeros_like(W)
    qmax = 2 ** bits - 1
    BL = GS * SB
    for b0 in range(0, IN, BL):
        b1 = b0 + BL
        W1 = W[:, b0:b1].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[b0:b1, b0:b1]
        p = gw._gw_one(W1, bits, GS, SB, -6, ex2[b0:b1] if ex2 is not None else None, True)
        sc = p["scale"].view(OUTd, -1); mn = p["mn"].view(OUTd, -1)
        for j in range(BL):
            g = j // GS
            w = W1[:, j]
            qv = torch.clamp(torch.round((w - mn[:, g]) / sc[:, g]), 0, qmax) * sc[:, g] + mn[:, g]
            Q[:, b0 + j] = qv
            e = (w - qv) / Hi[j, j]
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        W[:, b1:] -= E1 @ Hinv[b0:b1, b1:]
    return Q


# --- проверка 0: диагональная H -> ровно RTN
torch.manual_seed(0)
Wt = torch.randn(64, 512, device=DEV) * 0.02
ex = torch.rand(512) + 0.1
Qg = gptq(Wt, torch.diag(ex * 1000.0), 4, ex, damp=0.0)
Qr = gw._gw_one(Wt, 4, GS, SB, -6, ex, False)
print("проверка 0 (диагональная H == RTN): %s, макс |d| %.2e" % (torch.equal(Qg, Qr), (Qg - Qr).abs().max()), flush=True)
assert torch.equal(Qg, Qr)

# --- модель, конфиг, окна
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
if OVR:
    cfg.bits_overrides = dict(OVR, **cfg.bits_overrides)
stats = gw.load_act_stats(cfg.act_stats_path)
M = R.RWKV7Ref(CK, device=DEV, dtype=torch.bfloat16, compute_dtype=torch.float32)
enc = A._encoder(TOK)
text = open(A.CORPUS, encoding="utf-8").read()
import re
chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", text) if c.strip()]
cal = A._windows(chunks, enc, A.SEQ_LEN, NCAL * A.SEQ_LEN)[:NCAL]
cal = torch.tensor(cal, dtype=torch.long)[:, :-1].contiguous().to(DEV)
ev = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
cd = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_code_heldout.pt"))
Wv = torch.cat([ev["tokens"][:, :512], cd["tokens"]]).long()
langs = list(ev["lang"]) + list(cd["lang"]); kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"])
edata, etgt = Wv[:, :-1].contiguous().to(DEV), Wv[:, 1:].contiguous().to(DEV)
print("калибровка %d окон x %d, оценка %d окон; своп старт" % (cal.shape[0], cal.shape[1], len(langs)), flush=True)
BS = 8
kdt = torch.float32
head32 = M.head_weight.float()


def sync(t):
    # без синхронизации MPS копит сотни тысяч мелких запусков _wkv7 в одном командном
    # буфере и падает на выделении IOGPUDeviceShmem (0.1B, 27.09)
    torch.mps.synchronize()
    return t


def hidden(c):
    return torch.cat([sync(M.forward(edata[i:i + BS], cfg=c, return_hidden=True)) for i in range(0, edata.shape[0], BS)])


def stats_kl(h, head):
    kl, ce = [], []
    for j in range(h.shape[0]):
        lp = torch.log_softmax((href[j] @ head32.T).to(kdt), -1)
        lq = torch.log_softmax((h[j] @ head.T).to(kdt), -1)
        kl.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        ce.append(float(-lq.gather(-1, etgt[j][:, None]).mean()))
    return kl, ce


res = dict(ckpt=os.path.basename(CK), ncal=NCAL, damp=DAMP, overrides=bool(OVR), langs=langs, kind=kind, kl={}, ce={})
with torch.no_grad():
    href = hidden(None)
    res["ce"]["ref"] = stats_kl(href, head32)[1]
    t1 = time.time()
    res["kl"]["rtn"], res["ce"]["rtn"] = stats_kl(hidden(cfg), M._q(M.head_weight, "head", cfg, "head.weight"))
    json.dump(res, open(OUT, "w"))
    print("эталон и RTN: %.0f с (RTN %.0f с)" % (time.time() - t0, time.time() - t1), flush=True)

    # --- последовательный GPTQ
    done = {}                      # ключ -> 16 (веса уже заменены деквантом GPTQ)
    Hs = {}

    def rec(name, x):
        if name in Hs:
            v = x.detach().float().reshape(-1, x.shape[-1])
            Hs[name] += (v.T @ v).cpu()

    R._rec = rec

    def cseq():
        c = copy.copy(cfg); c.bits_overrides = dict(done, **cfg.bits_overrides); return c

    def collect(keys, fn):
        for k in keys:
            n = int(stats[k].numel())
            Hs[k] = torch.zeros(n, n)
        fn()
        out = {k: Hs.pop(k) for k in keys}
        return out

    def do(obj, attr, key, group, H):
        bits = ap._bits_of(cfg, group, key)
        if bits >= 16:
            return
        W = getattr(obj, attr).float()
        Q = gptq(W, H, bits, stats.get(key))
        setattr(obj, attr, Q)          # fp32: деквант без bf16-округления, как у RTN-пути
        done[key] = 16

    x = F.embedding(cal, M._q(M.emb_weight, "emb", cfg, "emb.weight"))
    x = F.layer_norm(x.float(), (M.n_embd,), M.ln0_w.float(), M.ln0_b.float())
    vf = torch.empty_like(x)
    tg = time.time()

    def att_pass(i, x, vf):
        outs, vfs = [], []
        for b in range(0, x.shape[0], BS):
            xb, vb = x[b:b + BS], vf[b:b + BS]
            xn = F.layer_norm(xb, (M.n_embd,), M.ln1_w[i].float(), M.ln1_b[i].float())
            a, v2 = M._tmix_forward(xn, vb, M.tmix[i], i, cseq())
            outs.append(sync(xb + a)); vfs.append(v2)
        return torch.cat(outs), torch.cat(vfs)

    def ffn_pass(i, x):
        outs = []
        for b in range(0, x.shape[0], BS):
            xb = x[b:b + BS]
            xn = F.layer_norm(xb, (M.n_embd,), M.ln2_w[i].float(), M.ln2_b[i].float())
            outs.append(sync(xb + M._cmix_forward(xn, M.cmix[i], cseq(), i)))
        return torch.cat(outs)

    for i in range(M.n_layer):
        t, c = M.tmix[i], M.cmix[i]
        p = "blocks.%d." % i
        H = collect([p + "att.receptance.weight", p + "att.key.weight", p + "att.value.weight"], lambda: att_pass(i, x, vf))
        for attr, nm in (("r_proj", "receptance"), ("k_proj", "key"), ("v_proj", "value")):
            do(t, attr, p + "att.%s.weight" % nm, "proj", H[p + "att.%s.weight" % nm])
        H = collect([p + "att.output.weight"], lambda: att_pass(i, x, vf))
        do(t, "o_proj", p + "att.output.weight", "proj", H[p + "att.output.weight"])
        x, vf = att_pass(i, x, vf)
        H = collect([p + "ffn.key.weight"], lambda: ffn_pass(i, x))
        do(c, "key", p + "ffn.key.weight", "cmix", H[p + "ffn.key.weight"])
        H = collect([p + "ffn.value.weight"], lambda: ffn_pass(i, x))
        do(c, "value", p + "ffn.value.weight", "cmix", H[p + "ffn.value.weight"])
        x = ffn_pass(i, x)
        print("[gptq] слой %d/%d, %.0f с" % (i + 1, M.n_layer, time.time() - tg), flush=True)
    xo = F.layer_norm(x, (M.n_embd,), M.ln_out_w.float(), M.ln_out_b.float()).reshape(-1, M.n_embd)
    Hh = (xo.T @ xo).cpu()
    del x, vf, xo
    Qh = gptq(M.head_weight.float(), Hh, ap._bits_of(cfg, "head", "head.weight"), stats.get("head.weight"))
    R._rec = lambda n, x: None
    cg = cseq()
    res["kl"]["gptq"], res["ce"]["gptq"] = stats_kl(hidden(cg), Qh)
    res["gptq_seconds"] = time.time() - tg
    json.dump(res, open(OUT, "w"))

ix = lambda pr: [j for j in range(len(langs)) if pr(j)]
def ppl(n, k):
    I = ix(lambda j: kind[j] == k); return math.exp(sum(res["ce"][n][j] for j in I) / len(I))
cols = [("текст", lambda j: kind[j] == "text"), ("en", lambda j: langs[j] == "en"), ("ru", lambda j: langs[j] == "ru"),
        ("sr", lambda j: langs[j] == "sr"), ("код", lambda j: kind[j] == "code"), ("py", lambda j: langs[j] == "py"), ("swift", lambda j: langs[j] == "swift")]
print("%-6s" % "" + "".join("%9s" % c for c, _ in cols) + "  Δppl текст  Δppl код")
for n in ("rtn", "gptq"):
    cells = []
    for _, pr in cols:
        I = ix(pr); a = sum(res["kl"][n][j] for j in I) / len(I); b = sum(res["kl"]["rtn"][j] for j in I) / len(I)
        cells.append("%9.5f" % a if n == "rtn" else "%+8.1f%%" % (100 * (a / b - 1)))
    print("%-6s" % n + "".join(cells) + "   %+.3f%%    %+.3f%%" % (100 * (ppl(n, "text") / ppl("ref", "text") - 1), 100 * (ppl(n, "code") / ppl("ref", "code") - 1)))
print("GPTQ %.0f с, всего %.0f с, своп %+.0f МБ" % (res["gptq_seconds"], time.time() - t0, sw() - s0), flush=True)
