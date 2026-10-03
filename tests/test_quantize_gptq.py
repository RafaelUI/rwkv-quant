"""Гейт GPTQ в writer (28.09), 0.1B g1d.

1. Части: при ДИАГОНАЛЬНОЙ H gptq_parts ПОБИТНО равны частям _gw_one (q, qs, qm, d, dm, deq) --
   bits 4/5/6 x режимы asym_sb6_aw / _search / asym_sb6, строка с нулевым суперблоком включена.
   Значит, GPTQ пишет в ТУ ЖЕ сетку, а отличие от RTN -- только компенсация ошибки.
2. Файл: calibration.gptq.run -> quantize_file(gptq_tensors=...) -> load_raw: каждый GPTQ-тензор
   декодируется ридером ПОБИТНО в deq GPTQ (bf16) -- файл несёт ровно то, что мерил fake-путь;
   прочие тензоры побайтно те же, что у файла без GPTQ; число GPTQ-матриц -- ожидаемое; манифест.
   Кернели Metal читают sb6 так же, как ридер (test_gw_dequant_kernel_parity) -- цепочка замкнута.
3. Качество (санитарно): на отложенном коде KL(GPTQ) < KL(RTN) через fake-путь RWKV7Ref.
4. (30.09) Корпус пакета: calib_tokens(None) разбирает data/gptq_calib.jsonl словарём модели РОВНО в окна, что
   мерились 28-30.09 (gptq_calib2_2809.pt), если оба файла на месте; иначе -- громкий ПРОПУСК.
6. (30.09; 01.10 + reduction) Умолчание: GPTQ зовётся неявно ровно для preset="compression"|"reduction" без своего config и с real_gw=True
   (G.run подменён заглушкой -- проверяется решение, а не счёт); без корпуса -- пропуск, а не отказ.
Части 2, 3, 5 -- для COMPRESSION (sb6) и (01.10) REDUCTION (sym).
5. (30.09) Разнесение по картам: RWKVQ_DEVICE2="cuda:0,cuda:1" -- GPTQ на нём ПОБИТНО как на RWKVQ_DEVICE.
С 30.09 часть 2 идёт через ПУБЛИЧНЫЙ путь: quantize(gptq=True, gptq_calib=окна, device=RWKVQ_DEVICE) против
fake-пути calibration.gptq.run с теми же окнами (детерминизм проверен побитно: gptq_run_freeze_3009).
    python tests/test_quantize_gptq.py            (RWKVQ_DEVICE=cuda:0 RWKVQ_DEVICE2=cuda:0,cuda:1 -- на сервере)"""
import copy, hashlib, json, os, re, sys, tempfile, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rwkv_quant import presets, api
from rwkv_quant.calibration import act_stats as A, groupwise as gw, gptq as G
from rwkv_quant.formats import writer as Wr, reader as Rd

CK = os.environ.get("RWKVQ_CKPT", os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
TOK = os.environ.get("RWKVQ_TOK", "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
NW = int(os.environ.get("RWKVQ_GPTQ_WINDOWS", "16"))
DEV = os.environ.get("RWKVQ_DEVICE") or None
DEV2 = os.environ.get("RWKVQ_DEVICE2") or None
CALIB_PT = os.path.expanduser(os.environ.get("RWKVQ_CALIB_PT", "~/Develop/WKV-kvant/gptq_calib2_2809.pt"))
PARTS = ("q", "qs", "qm", "d", "dm")


def part1():
    torch.manual_seed(zlib.crc32(b"gptq_diag"))
    W = torch.randn(48, 1024) * 0.02
    W[5, 256:512] = 0.0
    g = torch.Generator().manual_seed(zlib.crc32(b"ex2"))
    ex = torch.rand(1024, generator=g) + 0.1
    n = 0
    for bits in (4, 5, 6):
        for sbb, e in ((-6, ex), (-6, None), (6, None)):
            p = G.gptq_parts(W.clone(), torch.diag((e if e is not None else torch.ones(1024)) * 1000.0), bits, e, 0.0, 32, sbb)
            r = gw._gw_one(W.clone(), bits, 32, 8, sbb, e, True)
            for k in PARTS:
                a, b = p[k], r[k].view(p[k].shape) if r[k].numel() == p[k].numel() else r[k]
                assert torch.equal(a, b.to(a.dtype)), (bits, sbb, e is None, k)
            assert torch.equal(p["deq"], r["deq"]), (bits, sbb, "deq")
            n += 1
    m = 0
    for bits in (6, 8):                    # 01.10: sym (REDUCTION) -- части == _sym_one побитно
        for e, srch in ((ex, True), (None, True), (None, False)):
            p = G.gptq_parts_sym(W.clone(), torch.diag((e if e is not None else torch.ones(1024)) * 1000.0), bits, e, 0.0, 16, 16, srch)
            r = gw._sym_one(W.clone(), bits, 16, 16, e, srch, 0.0, True)
            for k in ("q", "qs", "d", "deq"):
                assert torch.equal(p[k], r[k].view(p[k].shape).to(p[k].dtype)), (bits, e is None, srch, k)
            m += 1
    print("1. части GPTQ при диагональной H == _gw_one (sb6) побитно: %d комбинаций; == _sym_one (sym): %d" % (n, m), flush=True)


def _windows():
    enc = A._encoder(TOK)
    chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", open(A.CORPUS, encoding="utf-8").read()) if c.strip()]
    return torch.tensor(A._windows(chunks, enc, A.SEQ_LEN, NW * A.SEQ_LEN)[:NW], dtype=torch.long)


def part2(preset="compression"):
    _, sig = A.collect(CK, TOK)
    cfg = copy.deepcopy(presets.COMPRESSION if preset == "compression" else presets.REDUCTION)
    cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
    cal = _windows()
    t0 = time.time()
    qts, deqs = G.run(CK, TOK, cfg, n_windows=NW, damp=G.DAMP, device=DEV, keep_deq=True, verbose=False, calib=cal)
    sd = torch.load(CK, map_location="cpu", mmap=True)
    L = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    exp = sum(1 for i in range(L) for m in ("att.receptance", "att.key", "att.value", "att.output", "ffn.key", "ffn.value")
              if not (i == 0 and m == "att.output")) + 1
    assert len(qts) == exp, (len(qts), exp)
    assert all(torch.isfinite(v).all() for v in deqs.values())
    print("2a. [%s] GPTQ (fake-путь, устройство %s): %d матриц (ожидалось %d), %.0f с" % (preset, DEV, len(qts), exp, time.time() - t0), flush=True)
    d = tempfile.mkdtemp()
    fg, fr = os.path.join(d, "g.rwkvq"), os.path.join(d, "r.rwkvq")
    nw0 = G.N_WINDOWS
    G.N_WINDOWS = NW                       # гейт -- на NW окнах; в библиотеке всегда 600
    try:
        api.quantize(CK, fg, preset=preset, tokenizer=TOK, autopick=False, gptq=True, gptq_calib=cal,
                     device=DEV, verbose=False)
    finally:
        G.N_WINDOWS = nw0
    api.quantize(CK, fr, preset=preset, tokenizer=TOK, autopick=False, gptq=False, verbose=False)
    cg, cr = Rd.load_raw(fg), Rd.load_raw(fr)
    ng = nsame = 0
    for k, qt in cg.tensors.items():
        if k in deqs:
            got = Rd._dequantize_one(qt, torch.float32)   # 03.10: reader по умолчанию fp16; сверка -- одним кастом fp32 -> bf16
            assert torch.isfinite(got).all(), k
            assert torch.equal(got.to(torch.bfloat16), deqs[k].to(torch.bfloat16)), k
            ng += 1
        else:
            a, b = qt, cr.tensors[k]
            for f, v in vars(a).items():
                w = getattr(b, f)
                assert (torch.equal(v, w) if isinstance(v, torch.Tensor) else v == w), (k, f)
            nsame += 1
    assert ng == len(deqs), (ng, len(deqs))
    diffs = sum(1 for k in deqs if not torch.equal(Rd._dequantize_one(cg.tensors[k]), Rd._dequantize_one(cr.tensors[k])))
    assert diffs == len(deqs), ("GPTQ-тензор совпал с RTN -- GPTQ не сработал?", diffs, len(deqs))
    from safetensors import safe_open
    with safe_open(fg, framework="pt") as f:
        man = json.loads(f.metadata()["rwkvq"])
    with safe_open(fr, framework="pt") as f:
        man_r = json.loads(f.metadata()["rwkvq"])
    g = man.get("gptq") or {}
    exp_sha = hashlib.sha1(cal.numpy().tobytes()).hexdigest()[:16]
    assert g.get("windows") == NW and g.get("damp") == G.DAMP and g.get("calib_sha") == exp_sha and \
        g.get("matrices") == len(deqs) and "gptq" not in man_r, (g, "gptq" in man_r)
    print("2b. [%s] quantize(gptq=True): %d GPTQ-тензоров декодируются ридером в deq fake-пути побитно; %d прочих побайтно "
          "как quantize(gptq=False); размеры %.2f / %.2f МБ; манифест gptq: %s" % (
              preset, ng, nsame, os.path.getsize(fg) / 1e6, os.path.getsize(fr) / 1e6, g), flush=True)
    assert os.path.getsize(fg) == os.path.getsize(fr) or abs(os.path.getsize(fg) - os.path.getsize(fr)) < 4096
    # неявное умолчание: compression без real_gw не зовёт GPTQ; без корпуса -- пропуск с причиной, а не отказ
    assert api._gptq_skip_reason(CK, DEV, cal) is None
    return cfg, deqs


def part4():
    if not (os.path.exists(G.CALIB_FILE) and os.path.exists(CALIB_PT)):
        print("4. ПРОПУСК: нет %s или %s" % (G.CALIB_FILE, CALIB_PT), flush=True)
        return
    tok, label = G.calib_tokens(TOK, None, G.N_WINDOWS)
    ref = torch.load(CALIB_PT, weights_only=False)["tokens"][:G.N_WINDOWS]
    assert torch.equal(tok, ref.long()), "корпус пакета разбирается не в те окна, что мерились"
    print("4. корпус пакета -> %d окон x %d == %s побитно (%s)" % (tok.shape[0], tok.shape[1], os.path.basename(CALIB_PT), label), flush=True)


def part6():
    calls = []
    run0, file0 = G.run, G.CALIB_FILE
    G.run = lambda *a, **k: calls.append(k.get("device")) or {}
    d = tempfile.mkdtemp()
    try:
        cases = [("compression", {}, 1), ("reduction", {}, 1), ("compression", dict(real_gw=False), 0),
                 ("reduction", dict(real_gw=False), 0), ("compression", dict(config=copy.deepcopy(presets.COMPRESSION)), 0),
                 ("reduction", dict(config=copy.deepcopy(presets.REDUCTION)), 0), ("reduction", dict(gptq=False), 0)]
        for i, (pr, kw, n) in enumerate(cases):
            calls.clear()
            api.quantize(CK, os.path.join(d, "%d.rwkvq" % i), preset=pr, tokenizer=TOK, autopick=False, verbose=False, **kw)
            assert len(calls) == n, (pr, kw, calls)
        G.CALIB_FILE = os.path.join(d, "нет.jsonl")
        calls.clear()
        api.quantize(CK, os.path.join(d, "x.rwkvq"), preset="compression", tokenizer=TOK, autopick=False, verbose=False)
        assert not calls
        try:
            api.quantize(CK, os.path.join(d, "y.rwkvq"), preset="compression", tokenizer=TOK, autopick=False,
                         verbose=False, gptq=True)
            raise AssertionError("явный gptq=True без корпуса обязан падать")
        except FileNotFoundError:
            pass
    finally:
        G.run, G.CALIB_FILE = run0, file0
    print("6. умолчание GPTQ: compression и reduction -- да; real_gw=False / свой config / gptq=False -- нет; "
          "без корпуса: неявный пропущен, явный -- отказ", flush=True)


def part5(cfg):
    if not DEV2:
        print("5. ПРОПУСК: RWKVQ_DEVICE2 не задан (разнесение по картам -- на сервере)", flush=True)
        return
    cal = _windows()
    h = lambda qt: hashlib.sha1(b"".join(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                                         for _, v in sorted(vars(qt).items()) if isinstance(v, torch.Tensor))).hexdigest()
    a = G.run(CK, TOK, cfg, n_windows=NW, damp=G.DAMP, device=DEV, verbose=False, calib=cal)
    b = G.run(CK, TOK, cfg, n_windows=NW, damp=G.DAMP, device=DEV2, verbose=False, calib=cal)
    bad = [k for k in a if h(a[k]) != h(b[k])]
    print("5. %s против %s: %d из %d тензоров иные %s" % (DEV, DEV2, len(bad), len(a), bad[:4]), flush=True)
    assert sorted(a) == sorted(b) and not bad


def part3(cfg, deqs, preset="compression"):
    from rwkv_quant.models import rwkv7_ref as R
    dev = G.devices(DEV)[0]
    M = R.RWKV7Ref(CK, device=dev, dtype=torch.bfloat16, compute_dtype=torch.float32)
    cd = torch.load(os.path.expanduser(os.environ.get("RWKVQ_EVAL_CODE", "~/Develop/WKV-kvant/eval_code_heldout.pt")))
    X = cd["tokens"][:8, :-1].contiguous().to(dev)
    with torch.no_grad():
        ref = torch.log_softmax(M.forward(X).float(), -1)
        kl = lambda lq: float((ref.exp() * (ref - lq)).sum(-1).mean())
        k_rtn = kl(torch.log_softmax(M.forward(X, cfg=cfg).float(), -1))
        for key, Q in deqs.items():
            if key == "head.weight":
                M.head_weight = Q.to(M.head_weight.device)
                continue
            i, rest = int(key.split(".")[1]), key.split(".", 2)[2]
            obj = M.tmix[i] if rest.startswith("att.") else M.cmix[i]
            attr = {"att.receptance.weight": "r_proj", "att.key.weight": "k_proj", "att.value.weight": "v_proj",
                    "att.output.weight": "o_proj", "ffn.key.weight": "key", "ffn.value.weight": "value"}[rest]
            setattr(obj, attr, Q.to(getattr(obj, attr).device))
        c = copy.copy(cfg)
        c.bits_overrides = dict({k: 16 for k in deqs}, **{k: v for k, v in cfg.bits_overrides.items() if k not in deqs})
        k_g = kl(torch.log_softmax(M.forward(X, cfg=c).float(), -1))
    print("3. [%s] KL к bf16 на 8 окнах отложенного кода: RTN %.5f, GPTQ %.5f (%+.1f%%)" % (preset, k_rtn, k_g, 100 * (k_g / k_rtn - 1)), flush=True)
    assert k_g < k_rtn


if __name__ == "__main__":
    part1()
    cfg, deqs = part2()
    part3(cfg, deqs)
    cfg_r, deqs_r = part2("reduction")     # 01.10: GPTQ в REDUCTION (sym)
    part3(cfg_r, deqs_r, "reduction")
    part4()
    part6()
    part5(cfg)
    part5(cfg_r)
    print("test_quantize_gptq: ЗЕЛЁНЫЙ")
