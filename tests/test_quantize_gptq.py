"""Гейт GPTQ в writer (28.09), 0.1B g1d.

1. Части: при ДИАГОНАЛЬНОЙ H gptq_parts ПОБИТНО равны частям _gw_one (q, qs, qm, d, dm, deq) --
   bits 4/5/6 x режимы asym_sb6_aw / _search / asym_sb6, строка с нулевым суперблоком включена.
   Значит, GPTQ пишет в ТУ ЖЕ сетку, а отличие от RTN -- только компенсация ошибки.
2. Файл: calibration.gptq.run -> quantize_file(gptq_tensors=...) -> load_raw: каждый GPTQ-тензор
   декодируется ридером ПОБИТНО в deq GPTQ (bf16) -- файл несёт ровно то, что мерил fake-путь;
   прочие тензоры побайтно те же, что у файла без GPTQ; число GPTQ-матриц -- ожидаемое; манифест.
   Кернели Metal читают sb6 так же, как ридер (test_gw_dequant_kernel_parity) -- цепочка замкнута.
3. Качество (санитарно): на отложенном коде KL(GPTQ) < KL(RTN) через fake-путь RWKV7Ref.
    python tests/test_quantize_gptq.py"""
import copy, json, os, sys, tempfile, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, groupwise as gw, gptq as G
from rwkv_quant.formats import writer as Wr, reader as Rd

CK = os.environ.get("RWKVQ_CKPT", os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
TOK = os.environ.get("RWKVQ_TOK", "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
NW = int(os.environ.get("RWKVQ_GPTQ_WINDOWS", "16"))
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
    print("1. части GPTQ при диагональной H == _gw_one побитно: %d комбинаций" % n, flush=True)


def part2():
    _, sig = A.collect(CK, TOK)
    cfg = copy.deepcopy(presets.COMPRESSION)
    cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
    t0 = time.time()
    qts, deqs = G.run(CK, TOK, cfg, n_windows=NW, keep_deq=True, verbose=False)
    sd = torch.load(CK, map_location="cpu", mmap=True)
    L = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    exp = sum(1 for i in range(L) for m in ("att.receptance", "att.key", "att.value", "att.output", "ffn.key", "ffn.value")
              if not (i == 0 and m == "att.output")) + 1
    assert len(qts) == exp, (len(qts), exp)
    print("2a. GPTQ: %d матриц (ожидалось %d), %.0f с" % (len(qts), exp, time.time() - t0), flush=True)
    d = tempfile.mkdtemp()
    fg, fr = os.path.join(d, "g.rwkvq"), os.path.join(d, "r.rwkvq")
    meta = dict(windows=NW, damp=0.01)
    Wr.quantize_file(CK, fg, cfg, verbose=False, gptq_tensors=dict(qts), gptq_meta=meta)
    Wr.quantize_file(CK, fr, cfg, verbose=False)
    cg, cr = Rd.load_raw(fg), Rd.load_raw(fr)
    ng = nsame = 0
    for k, qt in cg.tensors.items():
        if k in deqs:
            got = Rd._dequantize_one(qt)
            assert torch.equal(got.to(torch.bfloat16), deqs[k].to(torch.bfloat16)), k
            ng += 1
        else:
            a, b = qt, cr.tensors[k]
            for f, v in vars(a).items():
                w = getattr(b, f)
                assert (torch.equal(v, w) if isinstance(v, torch.Tensor) else v == w), (k, f)
            nsame += 1
    diffs = sum(1 for k in deqs if not torch.equal(Rd._dequantize_one(cg.tensors[k]), Rd._dequantize_one(cr.tensors[k])))
    assert diffs == len(deqs), ("GPTQ-тензор совпал с RTN -- GPTQ не сработал?", diffs, len(deqs))
    from safetensors import safe_open
    with safe_open(fg, framework="pt") as f:
        man = json.loads(f.metadata()["rwkvq"])
    with safe_open(fr, framework="pt") as f:
        man_r = json.loads(f.metadata()["rwkvq"])
    assert man.get("gptq") == meta and "gptq" not in man_r, (man.get("gptq"), "gptq" in man_r)
    print("2b. файл: %d GPTQ-тензоров декодируются ридером в deq GPTQ побитно; %d прочих побайтно как без GPTQ; "
          "размеры %.2f / %.2f МБ; манифест gptq: %s" % (ng, nsame, os.path.getsize(fg) / 1e6, os.path.getsize(fr) / 1e6,
                                                        man["gptq"]), flush=True)
    assert os.path.getsize(fg) == os.path.getsize(fr) or abs(os.path.getsize(fg) - os.path.getsize(fr)) < 4096
    return cfg, deqs


def part3(cfg, deqs):
    from rwkv_quant.models import rwkv7_ref as R
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
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
    print("3. KL к bf16 на 8 окнах отложенного кода: RTN %.5f, GPTQ %.5f (%+.1f%%)" % (k_rtn, k_g, 100 * (k_g / k_rtn - 1)), flush=True)
    assert k_g < k_rtn


if __name__ == "__main__":
    part1()
    cfg, deqs = part2()
    part3(cfg, deqs)
    print("test_quantize_gptq: ЗЕЛЁНЫЙ")
