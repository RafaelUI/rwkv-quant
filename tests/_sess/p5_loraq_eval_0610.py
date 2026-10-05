"""06.10, п. 5: цена LORA_Q="sep" (8-битные LoRA на декоде) прибором статьи -- KL по окну к bf16-чекпоинту (RWKV7Ref, счёт
fp32, как file_eval_0310) на eval_text_heldout (36 окон) и eval_code_open (48), по 511 предсказаний.
Плечи -- РЕАЛЬНЫЙ путь Metal на одном и том же файле, чередованием по окнам в одном процессе:
  none -- LoRA плотным деквантом файла (так считает префилл);  sep -- LORA_Q="sep", 8 бит (так считает декод T=1).
Плечо sep снимается ПРЕФИЛЛОМ с qm.LORA_Q_DECODE_ONLY=False: те же квантованные веса LoRA, что на декоде, но окно идёт одним
вызовом (приём tests/eval_lora_quant.py). Сверка приёма: KL(none || sep) здесь должен сойтись с KL(цельный || декод) из
p5_stream_0610 (1.5B reduction 4.4e-5 на трёх окнах).
Две фазы, по модели на процесс (законы 2, 13):
    python p5_loraq_eval_0610.py ref  <ckpt> <ref.pt>                 # скрытые bf16-чекпоинта после ln_out + голова
    python p5_loraq_eval_0610.py arms <файл.rwkvq> <ref.pt> <out.json>
    python p5_loraq_eval_0610.py report <out.json> ..."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
W = os.path.expanduser("~/Develop/WKV-kvant/")

def windows():
    ev = torch.load(W + "eval_text_heldout.pt", weights_only=False); co = torch.load(W + "eval_code_open.pt", weights_only=False)
    wins = torch.cat([ev["tokens"][:, :512], co["tokens"][:, :512]]).long()
    kind = ["text"] * len(ev["lang"]) + ["code_open"] * len(co["lang"]); langs = list(ev["lang"]) + list(co["lang"])
    assert wins.shape == (84, 512) and len(kind) == len(langs) == 84, (wins.shape, len(kind), len(langs))
    return wins, kind, langs

def swap():
    import subprocess
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split()[0]

MODE = sys.argv[1]
if MODE == "ref":
    from rwkv_quant.calibration import autopick as ap
    from rwkv_quant.models.rwkv7_ref import RWKV7Ref
    CK, OUT = sys.argv[2:4]; wins, kind, langs = windows(); t0 = time.time(); s0 = swap()
    M = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
    data = wins[:, :-1].contiguous(); I = ap._Instrument(M, data); hs = []
    with torch.no_grad():
        for i in range(0, 84, 4):
            hs.append(M.forward(data[i:i + 4], cfg=None, return_hidden=True).float().cpu()); print("  окна %d/84, %.0f с" % (i + 4, time.time() - t0), flush=True)
    torch.save(dict(h=torch.cat(hs), head=I.head32.float().cpu(), ckpt=os.path.basename(CK)), OUT)
    print("эталон %s -> %s (%.0f МБ), %.0f с, своп %s -> %s" % (os.path.basename(CK), OUT, os.path.getsize(OUT) / 1e6, time.time() - t0, s0, swap()))
elif MODE == "arms":
    import mlx.core as mx
    import rwkv_quant.backends.metal.quant_model as qm
    from rwkv_quant.formats.reader import load_raw
    F, REF, OUT = sys.argv[2:5]; wins, kind, langs = windows(); t0 = time.time(); s0 = swap()
    ref = torch.load(REF, weights_only=False); H, HEAD = ref["h"], ref["head"].double()
    m = qm.QuantRWKV7(load_raw(F)); qm.LORA_Q_DECODE_ONLY = False
    assert qm.LORA_QBITS == 8 and qm.LORA_Q == "sep", (qm.LORA_Q, qm.LORA_QBITS)
    rows = []
    for j in range(84):
        lp = torch.log_softmax(H[j].double() @ HEAD.T, -1); tgt = wins[j, 1:]
        ids = mx.array(wins[j:j + 1, :-1].numpy().astype(np.int32)); r = dict(j=j, kind=kind[j], lang=langs[j]); lqs = {}
        for name, mode in ((("none", None), ("sep", "sep")) if j % 2 == 0 else (("sep", "sep"), ("none", None))):   # чередование порядка плеч
            qm.LORA_Q = mode
            lg, _ = m.forward_stateful(ids, m.init_state(1)); lg = np.array(lg.astype(mx.float32))[0]
            assert np.isfinite(lg).all(), (j, name)
            lq = torch.log_softmax(torch.from_numpy(lg).double(), -1); lqs[name] = lq
            r["kl_" + name] = float((lp.exp() * (lp - lq)).sum(-1).mean()); r["ce_" + name] = float(-lq.gather(-1, tgt[:, None]).mean())
            r["top1_" + name] = float((lq.argmax(-1) == lp.argmax(-1)).float().mean())
        r["kl_none_sep"] = float((lqs["none"].exp() * (lqs["none"] - lqs["sep"])).sum(-1).mean())
        r["ce_ref"] = float(-lp.gather(-1, tgt[:, None]).mean())
        if j == 0:
            built = sum(b.tmix._lq_A is not None for b in m.blocks)
            assert built == len(m.blocks) and r["kl_none_sep"] > 0, "плечо sep не квантовано: %d слоёв, KL плеч %g" % (built, r["kl_none_sep"])
        rows.append(r)
        if j % 12 == 11: print("  окно %d/84: KL none %.6f sep %.6f, %.0f с" % (j + 1, r["kl_none"], r["kl_sep"], time.time() - t0), flush=True)
    qm.LORA_Q = "sep"; qm.LORA_Q_DECODE_ONLY = True
    json.dump(dict(file=F, ref=ref["ckpt"], rows=rows, swap=[s0, swap()]), open(OUT, "w"), ensure_ascii=False)
    print("плечи %s: %.0f с, своп %s -> %s -> %s" % (os.path.basename(F), time.time() - t0, s0, swap(), OUT))
else:
    rng = np.random.default_rng(20261006); B = 20000
    for p in sys.argv[2:]:
        d = json.load(open(p)); rows = d["rows"]
        print("\n%s (эталон %s), своп %s -> %s" % (os.path.basename(d["file"]), d["ref"], *d["swap"]))
        print("  %-10s %4s | %10s %10s | %8s %19s | %9s %19s | %10s | top1 none/sep" % ("набор", "окон", "KL none", "KL sep", "KL sep/none", "95% ДИ", "dppl sep-none", "95% ДИ, п.п.", "KL(none||sep)"))
        sets = [("text", lambda r: r["kind"] == "text"), ("code_open", lambda r: r["kind"] == "code_open")] + \
               [(l, (lambda l: lambda r: r["kind"] == "text" and r["lang"] == l)(l)) for l in ("en", "ru", "sr")]
        for name, f in sets:
            R = [r for r in rows if f(r)]; n = len(R)
            a = np.array([r["kl_none"] for r in R]); b = np.array([r["kl_sep"] for r in R])
            ca = np.array([r["ce_none"] for r in R]); cb = np.array([r["ce_sep"] for r in R]); cr = np.array([r["ce_ref"] for r in R])
            idx = rng.integers(0, n, size=(B, n))
            ratio = b[idx].mean(1) / a[idx].mean(1) - 1
            dppl = (np.exp(cb[idx].mean(1) - cr[idx].mean(1)) - np.exp(ca[idx].mean(1) - cr[idx].mean(1))) * 100
            print("  %-10s %4d | %10.6f %10.6f | %+7.2f%% [%+6.2f%%; %+6.2f%%] | %+8.4f [%+7.4f; %+7.4f] | %10.3e | %.4f / %.4f" % (
                name, n, a.mean(), b.mean(), 100 * (b.mean() / a.mean() - 1), 100 * np.percentile(ratio, 2.5), 100 * np.percentile(ratio, 97.5),
                (np.exp(cb.mean() - cr.mean()) - np.exp(ca.mean() - cr.mean())) * 100, np.percentile(dppl, 2.5), np.percentile(dppl, 97.5),
                np.mean([r["kl_none_sep"] for r in R]), np.mean([r["top1_none"] for r in R]), np.mean([r["top1_sep"] for r in R])))
