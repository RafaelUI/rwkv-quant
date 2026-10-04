"""04.10: публичный API rwkv-quant глазами нового пользователя, на 0.1B. Пакет берётся из УСТАНОВЛЕННОГО колеса
(PYTHONPATH=/tmp/rq_site, cwd=/tmp), не из репозитория. Каждая проверка -- отдельно, провал не останавливает остальные.
    cd /tmp && PYTHONPATH=/tmp/rq_site python api_smoke_0410.py [fast|full]   (full: ещё умолчания с GPTQ, ~40 мин)"""
import hashlib, json, os, subprocess, sys, time, traceback
MODE = sys.argv[1] if len(sys.argv) > 1 else "fast"
W = os.path.expanduser("~/Develop/WKV-kvant/"); CK = W + "rwkv7-g1d-0.1b.pth"
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
O = "/tmp/rq_smoke/"; os.makedirs(O, exist_ok=True)
RES = []
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def check(name):
    def deco(f):
        t0 = time.time()
        try:
            r = f(); RES.append((name, "OK", r, time.time() - t0))
        except Exception as e:
            RES.append((name, "ПРОВАЛ", "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:200]), time.time() - t0)); traceback.print_exc()
        print("[%s] %s -- %s (%.0f с)" % (RES[-1][1], name, RES[-1][2], RES[-1][3]), flush=True)
        return f
    return deco

@check("01 ленивый импорт: import rwkv_quant и codec без torch")
def _():
    r = subprocess.run([sys.executable, "-c", "import sys, rwkv_quant; from rwkv_quant.formats import codec; assert 'torch' not in sys.modules; print(rwkv_quant.__file__, sorted(rwkv_quant.__all__))"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    assert "/tmp/rq_site" in r.stdout, r.stdout
    return r.stdout.strip()

import torch
import rwkv_quant
from rwkv_quant import quantize, calibrate, QuantConfig, GROUPS
from rwkv_quant.formats import codec
from rwkv_quant.formats.reader import load_raw, load_dequantized
assert "/tmp/rq_site" in rwkv_quant.__file__, rwkv_quant.__file__
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
ev = torch.load(W + "eval_text_heldout.pt", weights_only=False)["tokens"]
WIN = ev[[0, 12, 24], :257].contiguous()      # en / ru / sr
REF = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
with torch.no_grad(): LREF = torch.log_softmax(REF.forward(WIN[:, :-1], cfg=None).float(), -1)
def kl_file(p):
    dq = load_dequantized(p); sd = torch.load(CK, map_location="cpu", mmap=True, weights_only=True); out = {}
    for k, v in sd.items():
        w = dq[k]
        if tuple(w.shape) != tuple(v.shape): w = w.T if (w.dim() == 2 and tuple(w.T.shape) == tuple(v.shape)) else w.reshape(v.shape)
        out[k] = w.float().contiguous()
    tmp = O + "deq.pth"; torch.save(out, tmp)
    M = RWKV7Ref(tmp, device="cpu", dtype=torch.float32, compute_dtype=torch.float32)
    with torch.no_grad(): lq = torch.log_softmax(M.forward(WIN[:, :-1], cfg=None).float(), -1)
    os.remove(tmp)
    return float((LREF.exp() * (LREF - lq)).sum(-1).mean())
def info(p):
    man, arr = codec.open_rwkvq(p)
    return man, "%.1f МБ, KL %.5f" % (os.path.getsize(p) / 1e6, kl_file(p))

@check("02 quantize без токенизатора -> понятная ошибка")
def _():
    try: quantize(CK, O + "x.rwkvq", verbose=False, gptq=False)
    except Exception as e: return "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:120])
    raise AssertionError("не упал: файл записан без токенизатора")

@check("03 reduction, gptq=False, токенизатор = путь к словарю")
def _():
    quantize(CK, O + "red.rwkvq", preset="reduction", tokenizer=TOK, gptq=False, verbose=False)
    man, s = info(O + "red.rwkvq"); return s + "; ключи манифеста: " + ", ".join(sorted(man)[:14])

@check("04 токенизатор = callable и объект с .encode -> тот же файл")
def _():
    from rwkv_quant.calibration import act_stats as A
    enc = A._encoder(TOK)
    class T:
        def encode(self, s): return enc(s)
    quantize(CK, O + "red_c.rwkvq", tokenizer=enc, gptq=False, verbose=False)
    quantize(CK, O + "red_o.rwkvq", tokenizer=T(), gptq=False, verbose=False)
    a, b, c = md5(O + "red.rwkvq"), md5(O + "red_c.rwkvq"), md5(O + "red_o.rwkvq")
    assert a == b == c, (a, b, c)
    return "md5 одинаков у трёх способов (и preset по умолчанию == reduction)"

@check("05 compression, gptq=False, autopick по умолчанию")
def _():
    quantize(CK, O + "comp_ap.rwkvq", preset="compression", tokenizer=TOK, gptq=False, verbose=False)
    man, s = info(O + "comp_ap.rwkvq"); return s + "; autopick в манифесте: %s" % (str(man.get("autopick"))[:160])

@check("06 compression, autopick=False и autopick_budget=0")
def _():
    quantize(CK, O + "comp.rwkvq", preset="compression", tokenizer=TOK, gptq=False, autopick=False, verbose=False)
    quantize(CK, O + "comp_b0.rwkvq", preset="compression", tokenizer=TOK, gptq=False, autopick_budget=0, verbose=False)
    s0, s1, s2 = (os.path.getsize(O + n) for n in ("comp.rwkvq", "comp_b0.rwkvq", "comp_ap.rwkvq"))
    assert s1 <= s0, (s0, s1); assert s2 <= s0 * 1.005 + 1, (s0, s2)
    return "пресет %s | бюджет 0: %+.3f%% байт, %s | бюджет 0.5%%: %+.3f%% байт" % (info(O + "comp.rwkvq")[1], 100 * (s1 / s0 - 1), info(O + "comp_b0.rwkvq")[1], 100 * (s2 / s0 - 1))

@check("07 act_stats=None (без AW-статистики)")
def _():
    quantize(CK, O + "red_noaw.rwkvq", tokenizer=None, act_stats=None, gptq=False, verbose=False)
    return info(O + "red_noaw.rwkvq")[1]

@check("08 config=QuantConfig(...) вручную, как в docstring: только биты")
def _():
    quantize(CK, O + "cfg_bits.rwkvq", tokenizer=TOK, config=QuantConfig(proj=4, cmix=4, emb=6, head=6), verbose=False)
    return info(O + "cfg_bits.rwkvq")[1]

@check("09 config вручную с group_scale / режимами (копия пресета, proj=6)")
def _():
    import copy
    from rwkv_quant import presets
    c = copy.deepcopy(presets.REDUCTION); c.bits["proj"] = 6
    quantize(CK, O + "cfg_full.rwkvq", tokenizer=TOK, config=c, verbose=False)
    man, s = info(O + "cfg_full.rwkvq"); return s + "; gptq в манифесте: %s" % (str(man.get("gptq"))[:80])

@check("10 gptq=True со своим gptq_calib (8 окон)")
def _():
    quantize(CK, O + "red_g8.rwkvq", tokenizer=TOK, gptq=True, gptq_calib=ev[36 - 8:, :512].contiguous(), verbose=False)
    man, s = info(O + "red_g8.rwkvq"); return s + "; gptq в манифесте: %s" % (str(man.get("gptq"))[:200])

@check("11 читалки: load_dequantized (fp16) против codec.dequant_key (numpy, без torch)")
def _():
    import numpy as np
    dq = load_dequantized(O + "red.rwkvq"); man, arr = codec.open_rwkvq(O + "red.rwkvq")
    keys = ["emb.weight", "head.weight", "blocks.0.att.output.weight", "blocks.3.att.key.weight", "blocks.5.ffn.value.weight", "blocks.2.att.w1", "blocks.1.ln1.weight"]
    bad, dts = [], set()
    for k in keys:
        a = dq[k]; dts.add(str(a.dtype)); b = codec.dequant_key(man, arr, k)
        bt = torch.from_numpy(np.ascontiguousarray(b))
        if bt.numel() != a.numel(): bad.append((k, tuple(bt.shape), tuple(a.shape))); continue
        if tuple(bt.shape) != tuple(a.shape): bt = bt.T if bt.dim() == 2 and tuple(bt.T.shape) == tuple(a.shape) else bt.reshape(a.shape)
        d = float((bt.float().to(a.dtype).float() - a.float()).abs().max())
        if d != 0: bad.append((k, d))
    assert not bad, bad
    return "7 ключей совпали побитно после приведения к типу читалки; типы читалки %s; всего тензоров %d" % (sorted(dts), len(dq))

@check("12 инференс Metal: step (префилл + декод), состояние, generate")
def _():
    import mlx.core as mx, numpy as np
    from rwkv_quant.backends.metal.quant_model import QuantRWKV7
    from rwkv_quant.backends.metal import generate as G
    out = []
    for name in ("red.rwkvq", "comp_ap.rwkvq"):
        m = QuantRWKV7(load_raw(O + name))
        ids = mx.array(WIN[:1, :-1].numpy().astype(np.int32))
        lg, st = m.step(ids, m.init_state(1)); lg = np.array(lg.astype(mx.float32))
        lq = torch.log_softmax(torch.from_numpy(lg), -1)
        kl = float((LREF[:1].exp() * (LREF[:1] - lq)).sum(-1).mean())
        # префилл 128 + 128 шагов по одному == префилл 256 (последний логит)
        l1, s1 = m.step(ids[:, :128], m.init_state(1))
        for t in range(128, 256): l1, s1 = m.step(ids[:, t:t + 1], s1)
        d = float(np.abs(np.array(l1[0, -1].astype(mx.float32)) - lg[0, -1]).max())
        toks, _ = G.generate(m, WIN[0, :64].tolist(), 24)
        toks2, _ = G.generate(m, WIN[0, :64].tolist(), 24, pipeline=False)
        assert toks == toks2, "generate: конвейер и синхронный путь разошлись"
        assert kl < 0.2 and d < 0.5, (kl, d)
        out.append("%s: KL к bf16 (окно en) %.5f, потоковый против цельного max|dlogit| %.3f, generate 24 ток. совпал в двух режимах" % (name, kl, d))
    return " | ".join(out)

@check("13 calibrate() на одной группе -> конфиг -> quantize(config=)")
def _():
    cfg = calibrate(CK, W + "eval_text_heldout.pt", device="mps", groups=["cmix"], n_seq=4, seq_len=256, verbose=False)
    quantize(CK, O + "calib.rwkvq", tokenizer=TOK, config=cfg, verbose=False)
    return "cmix -> %s бит, режим %s; файл %s" % (cfg.bits.get("cmix"), (cfg.group_scale_mode or {}).get("cmix"), info(O + "calib.rwkvq")[1])

@check("14 export_mlx.export: сайдкар safetensors")
def _():
    from rwkv_quant.formats import export_mlx
    export_mlx.export(O + "red.rwkvq", O + "red_sidecar", tokenizer=TOK)
    fs = sorted(os.listdir(O)); return "создано: %s" % [f for f in fs if f.startswith("red_sidecar")]

if MODE == "full":
    @check("15 УМОЛЧАНИЯ: quantize(ckpt, out, tokenizer=tok) -- reduction + GPTQ 600 окон")
    def _():
        quantize(CK, O + "red_default.rwkvq", tokenizer=TOK, verbose=False)
        man, s = info(O + "red_default.rwkvq"); return s + "; gptq: %s" % (str(man.get("gptq"))[:160])
    @check("16 УМОЛЧАНИЯ: preset=compression -- autopick + GPTQ")
    def _():
        quantize(CK, O + "comp_default.rwkvq", preset="compression", tokenizer=TOK, verbose=False)
        man, s = info(O + "comp_default.rwkvq"); return s + "; gptq: %s; autopick: %s" % (str(man.get("gptq"))[:100], str(man.get("autopick"))[:100])

print("\n== ИТОГ: %d проверок, провалов %d" % (len(RES), sum(r[1] != "OK" for r in RES)))
for n, s, r, t in RES: print("%-7s %-70s %5.0f с | %s" % (s, n, t, r))
