"""06.10, находка 1 проверки API: при каких сочетаниях полей QuantConfig quantize(config=...) пишет сломанный файл.
0.1B, установленное колесо (PYTHONPATH=/tmp/rq_site_p1, cwd=/tmp), один процесс, файл перезаписывается.
KL(bf16-эталон с fp32-активациями || файл) на трёх окнах en / ru / sr по 256 токенов -- тот же прибор, что api_smoke_0410.
    cd /tmp && PYTHONPATH=/tmp/rq_site_p1 python ~/Develop/rwkv-quant/tests/_sess/p1_cfg_grid_0610.py <журнал.json>"""
import copy, glob, json, os, sys, time, warnings
import torch
torch.set_num_threads(4)
import rwkv_quant
from rwkv_quant import quantize, QuantConfig, presets
from rwkv_quant.formats.reader import load_dequantized
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
assert "/tmp/rq_site_p1" in rwkv_quant.__file__, rwkv_quant.__file__
OUTJ = sys.argv[1]
W = os.path.expanduser("~/Develop/WKV-kvant/"); CK = W + "rwkv7-g1d-0.1b.pth"
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
O = "/tmp/rq_p1/"; os.makedirs(O, exist_ok=True); F = O + "x.rwkvq"
ev = torch.load(W + "eval_text_heldout.pt", weights_only=False)["tokens"]
WIN = ev[[0, 12, 24], :257].contiguous()
REF = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
with torch.no_grad(): LREF = torch.log_softmax(REF.forward(WIN[:, :-1], cfg=None).float(), -1)
SD = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)
def kl_file(p):
    dq = load_dequantized(p); out = {}
    for k, v in SD.items():
        w = dq[k]
        if tuple(w.shape) != tuple(v.shape): w = w.T if (w.dim() == 2 and tuple(w.T.shape) == tuple(v.shape)) else w.reshape(v.shape)
        out[k] = w.float().contiguous()
    tmp = O + "deq.pth"; torch.save(out, tmp)
    M = RWKV7Ref(tmp, device="cpu", dtype=torch.float32, compute_dtype=torch.float32)
    with torch.no_grad(): lq = torch.log_softmax(M.forward(WIN[:, :-1], cfg=None).float(), -1)
    os.remove(tmp)
    kl = float((LREF.exp() * (LREF - lq)).sum(-1).mean())
    top1 = float((LREF.argmax(-1) == lq.argmax(-1)).float().mean())
    return kl, top1
RES = []
def run(name, cfg=None, **kw):
    t0 = time.time(); rec = dict(name=name)
    if os.path.exists(F): os.remove(F)
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            if cfg is None: quantize(CK, F, tokenizer=TOK, verbose=False, **kw)
            else: quantize(CK, F, tokenizer=TOK, config=cfg, verbose=False, **kw)
        kl, top1 = kl_file(F)
        rec.update(kl=kl, top1=top1, mb=os.path.getsize(F) / 1e6, warnings=[str(w.message)[:160] for w in ws])
    except Exception as e:
        rec.update(error="%s: %s" % (type(e).__name__, str(e).split("\n")[0][:200]))
    rec["sec"] = time.time() - t0; RES.append(rec)
    print("%-46s %s" % (name, ("KL %.5f  top1 %.3f  %.1f МБ  предупр. %d" % (rec["kl"], rec["top1"], rec["mb"], len(rec["warnings"]))) if "kl" in rec else rec["error"]), "(%.0f с)" % rec["sec"], flush=True)
    json.dump(RES, open(OUTJ, "w"), ensure_ascii=False, indent=1)

# 0. опорные точки: пресеты без GPTQ и без автоподбора, плотный файл
run("bf16 (QuantConfig() пустой)", QuantConfig())
run("preset reduction, gptq=False", preset="reduction", gptq=False)
run("preset compression, gptq=False, autopick=False", preset="compression", gptq=False, autopick=False)
# 1. пример из docstring и его соседи
run("DOCSTRING proj=4 cmix=4 emb=6 head=6", QuantConfig(proj=4, cmix=4, emb=6, head=6))
run("все четыре @8 построчно", QuantConfig(proj=8, cmix=8, emb=8, head=8))
run("все четыре @6 построчно", QuantConfig(proj=6, cmix=6, emb=6, head=6))
run("все четыре @5 построчно", QuantConfig(proj=5, cmix=5, emb=5, head=5))
# 2. одна группа построчно, остальное bf16
for g in ("proj", "cmix", "emb", "head"):
    for b in (8, 6, 5, 4):
        run("только %s=%d построчно" % (g, b), QuantConfig(**{g: b}))
for g in ("w_lora", "a_lora", "v_lora", "g_lora"):
    for b in (8, 6, 4):
        run("только %s=%d построчно" % (g, b), QuantConfig(**{g: b}))
for b in (8, 6): run("только small=%d построчно" % b, QuantConfig(small=b))
# 3. те же биты, что в docstring, но с раскладкой пресета (что дала бы автоподстановка)
C = presets.COMPRESSION
run("docstring-биты + group_scale/режимы COMPRESSION", QuantConfig(proj=4, cmix=4, emb=6, head=6,
    group_scale=dict(C.group_scale), group_scale_mode=dict(C.group_scale_mode)))
run("docstring-биты + group_scale COMPRESSION, режим asym_sb6 (без AW, без статистики)", QuantConfig(proj=4, cmix=4, emb=6, head=6,
    group_scale={"proj": 32, "cmix": 32, "emb_head": 32}, group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6", "emb_head": "asym_sb6"}))
# 4. построчно со SpQR-выбросами и построчно с AW-статистикой (две другие построчные ветки писателя)
run("proj=4 cmix=4 построчно + outlier_fracs 2%", QuantConfig(proj=4, cmix=4, outlier_fracs={"proj": 0.02, "cmix": 0.02}))
run("proj=8 cmix=8 построчно + outlier_fracs 2%/1% (конфиг гейтов)", QuantConfig(proj=8, cmix=8, outlier_fracs={"proj": 0.02, "cmix": 0.01}))
from rwkv_quant.calibration import act_stats as A
st, sig = A.collect(CK, TOK, verbose=False); del st
AP = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
run("docstring-биты построчно + act_stats=путь (взвешенный RTN)", QuantConfig(proj=4, cmix=4, emb=6, head=6), act_stats=AP)
# 5. group_scale без режима (умолчание asym) и недопустимые пары режим/биты -- громко ли падает
run("group_scale proj=32 без режима, proj=4", QuantConfig(proj=4, group_scale={"proj": 32}))
run("group_scale proj=32 без режима, proj=6", QuantConfig(proj=6, group_scale={"proj": 32}))
run("asym_sb6_aw proj=8 (вне 4/5/6)", QuantConfig(proj=8, group_scale={"proj": 32}, group_scale_mode={"proj": "asym_sb6_aw"}))
run("sym_aw proj=4 (вне 6/8)", QuantConfig(proj=4, group_scale={"proj": 16}, group_scale_mode={"proj": "sym_aw"}))
run("режим без group_scale: proj=4, mode asym_sb6_aw", QuantConfig(proj=4, group_scale_mode={"proj": "asym_sb6_aw"}))
run("опечатка в группе: prooj=4", None) if False else None
try:
    QuantConfig(prooj=4); print("QuantConfig(prooj=4): принят молча, bits=%s" % QuantConfig(prooj=4).bits)
except Exception as e: print("QuantConfig(prooj=4): %s: %s" % (type(e).__name__, e))
print("ГОТОВО")
