"""Наши пресеты против схемы MLX-файлов MollySophia (mollysama/rwkv-mobile-models/mlx) на
ОДНОМ нашем чекпоинте, ОДНИМ прибором (22.09 ночь).
Их схема (прочитана из config.json их архивов): affine, группа 64, fp16 scale+bias
  mlx6: ВСЁ в 6 бит -- r/k/v/o, ffn.key/value, эмбеддинг, голова, LoRA «вниз» (w1/a1/v1/g1);
  mlx4: r/k/v/o и ffn в 4 бита; эмбеддинг, голова, LoRA «вниз» в 6 бит;
  остальное (LoRA «вверх», нормы, x_*, k_k...) -- fp16. Квантование -- сам mx.quantize/
  mx.dequantize на fp16-весах (ровно то, что лежит у них в файле), без AW.
Наши: fake-путь fake_quant.q (равен файлу, закон 39 / 22.09) -- reduction, compression и
  compression + autopick (json с выбором).
Прибор: RWKV7Ref, bf16-хранение + fp32-счёт, MPS; эталон -- тот же прибор без квантования;
KL по окну, все 38 окон eval_corpus_multiling (ни одно не участвовало в калибровке/выборе).
Сравнивается КВАНТОВАНИЕ ВЕСОВ, не их рантайм (он считает в fp16).
    python vs_mlx_affine_2209.py <ckpt> <act_stats.pt> <out.json> [autopick.json ...]"""
import copy, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import mlx.core as mx
from rwkv_quant import presets
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.calibration import autopick as ap
CK, ACT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EXTRA = sys.argv[4:]
DEV = "mps"
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split("M")[0])
blob = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
data = blob["tokens"][:, :512][:, :-1].contiguous().to(DEV); langs = list(blob["lang"])
tgt = blob["tokens"][:, 1:512].contiguous().to(DEV)   # следующие токены: для ppl (23.09)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
s0 = sw(); t0 = time.time()


def new():
    return RWKV7Ref(CK, device=DEV, dtype=torch.bfloat16, compute_dtype=torch.float32)


def hidden(M, cfg=None):
    with torch.no_grad():
        return torch.cat([M.forward(data[i:i + 8], cfg=cfg, return_hidden=True) for i in range(0, data.shape[0], 8)])


M = new()
h_ref = hidden(M); head_ref = M.head_weight.float(); L = M.n_layer
del M; torch.mps.empty_cache()
print("эталон %.0f с" % (time.time() - t0), flush=True)


CE_REF = []


def kl_per(h, head, extra=None):
    """KL по окну; в extra -- CE по окну (для ppl) и доля совпадения top-1 с эталоном."""
    out = []
    with torch.no_grad():
        for j in range(h.shape[0]):
            lp = torch.log_softmax(h_ref[j] @ head_ref.T, -1); lq = torch.log_softmax(h[j] @ head.T, -1)
            out.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
            if extra is not None:
                extra.setdefault("ce", []).append(float(-lq.gather(-1, tgt[j][:, None]).mean()))
                extra.setdefault("top1", []).append(float((lq.argmax(-1) == lp.argmax(-1)).float().mean()))
                if len(CE_REF) < h.shape[0]:
                    CE_REF.append(float(-lp.gather(-1, tgt[j][:, None]).mean()))
    return out


def mlx_q(w, bits):
    a = mx.array(w.to(torch.float16).cpu().numpy())
    wq, s, b = mx.quantize(a, group_size=64, bits=bits)
    d = mx.dequantize(wq, s, b, group_size=64, bits=bits)
    return torch.from_numpy(np.array(d, copy=False).astype(np.float16)).to(w.device)


def mlx_arm(M, bits_main):
    """(байты, число квантованных)"""
    byt, n = 0.0, 0
    def put(obj, attr, bits):
        nonlocal byt, n
        w = getattr(obj, attr)
        if w is None:
            return
        if bits:
            setattr(obj, attr, mlx_q(w, bits)); byt += w.numel() * (bits + 0.5) / 8; n += 1
        else:
            setattr(obj, attr, w.to(torch.float16)); byt += w.numel() * 2
    put(M, "emb_weight", 6); put(M, "head_weight", 6)
    for a in ("ln0_w", "ln0_b", "ln_out_w", "ln_out_b"):
        put(M, a, 0)
    for lst in (M.ln1_w, M.ln1_b, M.ln2_w, M.ln2_b):
        for i in range(len(lst)):
            byt += lst[i].numel() * 2; lst[i] = lst[i].to(torch.float16)
    for i, (t, c) in enumerate(zip(M.tmix, M.cmix)):
        for attr in ("r_proj", "k_proj", "v_proj", "o_proj"):
            put(t, attr, bits_main)
        for attr in ("w_lora_A", "a_lora_A", "g_lora_A", "v_lora_A"):
            put(t, attr, 6)
        for attr in type(t).__slots__:
            v = getattr(t, attr, None)
            if torch.is_tensor(v) and v.dtype == torch.bfloat16:
                put(t, attr, 0)
        put(c, "key", bits_main); put(c, "value", bits_main); put(c, "x_k", 0)
    return byt, n


def record(name, h, head, byt, note):
    ex = {}; kl = kl_per(h, head, ex); tot = float(np.mean(kl))
    ppl, ppl0 = float(np.exp(np.mean(ex["ce"]))), float(np.exp(np.mean(CE_REF)))
    res[name] = dict(kl=kl, mean=tot, bytes=byt, note=note, ce=ex["ce"], ce_ref=CE_REF, top1=ex["top1"],
                     ppl=ppl, ppl_ref=ppl0, dppl_pct=100 * (ppl / ppl0 - 1))
    json.dump(res, open(OUT, "w"), indent=1)
    print("%-22s ppl %.4f (эталон %.4f, %+.3f%%)  top-1 %.2f%%" % (name, ppl, ppl0, 100 * (ppl / ppl0 - 1), 100 * np.mean(ex["top1"])), flush=True)
    by = {g: float(np.mean([k for k, l in zip(kl, langs) if l == g])) for g in ("en", "ru", "sr")}
    print("%-22s KL %.5f  en %.5f ru %.5f sr %.5f  %8.1f МБ  (%.0f с, своп %+0.0f МБ)" % (
        name, tot, by["en"], by["ru"], by["sr"], byt / 1e6, time.time() - t0, sw() - s0), flush=True)


for bits in [int(b) for b in os.environ.get("VSMLX_BITS", "6,4").split(",") if b]:
    name = "mlx%d" % bits
    if name in res:
        continue
    M = new(); byt, n = mlx_arm(M, bits)
    h = hidden(M); record(name, h, M.head_weight.float(), byt, "их схема, квантовано %d матриц" % n)
    del M, h; torch.mps.empty_cache()

cfgs = [("reduction", presets.REDUCTION, {}), ("compression", presets.COMPRESSION, {})]
for f in EXTRA:
    cfgs.append((os.path.basename(f).replace(".json", ""), presets.COMPRESSION, json.load(open(f))))
for name, base, ovr in cfgs:
    if name in res:
        continue
    cfg = copy.deepcopy(base); cfg.act_stats_path = ACT
    if ovr:
        cfg.bits_overrides = dict(ovr, **{k: v for k, v in cfg.bits_overrides.items() if k not in ovr})
    M = new()
    byt = sum(ap.nbytes(getattr(o, a).numel(), ap._bits_of(cfg, g, k)) for o, a, g, k in ap._file_points(M))
    h = hidden(M, cfg); head = M._q(M.head_weight, "head", cfg, "head.weight")
    record(name, h, head, byt, "наш пресет, fake-путь" + (", ключей правила %d" % len(ovr) if ovr else ""))
    del M, h; torch.mps.empty_cache()
print("ГОТОВО за %.0f с" % (time.time() - t0))
