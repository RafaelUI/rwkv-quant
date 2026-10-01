"""01.10: ppl на РЕАЛЬНОМ пути (Metal, записанный .rwkvq) на ЧИСТЫХ отложенных наборах -- для README.
Окна: eval_text_heldout (36: Википедия en/ru/sr, 0 общих 16-грамм с calib_corpus и калибровкой GPTQ) + eval_code_heldout
(24: py/swift; закрытый код -- набор не публикуется). База -- плотная модель RWKV7Ref (bf16-хранение, fp32-счёт, MPS),
как у прежних строк README (fp32-эталон). Пер-окно CE -> парный бутстрэп по окнам.
Почему так, а не через ce_real_path_1909: там база -- сохранённые fp32-логиты (8 ГБ на модель для 60 окон); здесь база --
CE эталона на тех же окнах, логиты не храним. KL для README -- с сервера (fake-путь == файл побитно по кодам, гейт).
Один файл на процесс (законы 11 и 13).
    python real_ppl_heldout_0110.py --ref <ckpt> <метка>
    python real_ppl_heldout_0110.py <файл.rwkvq> <метка> <плечо>
    python real_ppl_heldout_0110.py --report"""
import json, math, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import torch
OUT = os.path.expanduser("~/Develop/WKV-kvant/real_ppl_heldout_0110.json")
W = os.path.expanduser("~/Develop/WKV-kvant/")


def windows():
    ev = torch.load(W + "eval_text_heldout.pt", weights_only=False); cd = torch.load(W + "eval_code_heldout.pt", weights_only=False)
    toks = ev["tokens"][:, :512].tolist() + cd["tokens"][:, :512].tolist()
    langs = list(ev["lang"]) + list(cd["lang"])
    kind = ["text"] * len(ev["lang"]) + ["code"] * len(cd["lang"])
    return np.array(toks, dtype=np.int64), langs, kind


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split()[0]


def save(lab, arm, ce, extra):
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    doc.setdefault(lab, {})[arm] = dict(ce=ce, **extra)
    json.dump(doc, open(OUT, "w"), indent=0)


def main():
    data, langs, kind = windows()
    t0 = time.time(); s0 = swap()
    if sys.argv[1] == "--ref":
        from rwkv_quant.models.rwkv7_ref import RWKV7Ref
        ck, lab = sys.argv[2], sys.argv[3]
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        M = RWKV7Ref(ck, device=dev, dtype=torch.bfloat16, compute_dtype=torch.float32)
        ce = []
        with torch.no_grad():
            for i in range(0, len(data), 4):
                X = torch.tensor(data[i:i + 4, :-1], device=dev); Y = torch.tensor(data[i:i + 4, 1:], device=dev)
                lp = torch.log_softmax(M.forward(X).float(), -1)
                ce += [float(v) for v in -lp.gather(-1, Y[..., None])[..., 0].mean(-1).cpu()]
                del lp
        arm, extra = "ref", dict(ckpt=os.path.basename(ck))
    else:
        import mlx.core as mx
        from rwkv_quant.formats.reader import load_raw
        from rwkv_quant.backends.metal.quant_model import QuantRWKV7
        path, lab, arm = sys.argv[1:4]
        model = QuantRWKV7(load_raw(path))
        ce = []
        for i in range(len(data)):
            lg = model(mx.array(data[i:i + 1, :-1])).astype(mx.float32); mx.eval(lg)
            x = np.array(lg)[0].astype(np.float64); del lg
            m = x.max(1, keepdims=True); lse = (m + np.log(np.exp(x - m).sum(1, keepdims=True)))[:, 0]
            ce.append(float((lse - x[np.arange(511), data[i, 1:]]).mean()))
        extra = dict(file=os.path.basename(path), bytes=os.path.getsize(path))
    extra.update(langs=langs, kind=kind, seconds=round(time.time() - t0), swap_before=s0, swap_after=swap())
    save(lab, arm, ce, extra)
    print("%s %s: ppl текст %.4f код %.4f, %.0f с, своп %s -> %s" % (lab, arm, math.exp(np.mean([c for c, k in zip(ce, kind) if k == "text"])),
          math.exp(np.mean([c for c, k in zip(ce, kind) if k == "code"])), time.time() - t0, s0, extra["swap_after"]), flush=True)


def report():
    doc = json.load(open(OUT)); rng = np.random.default_rng(20261001)
    for lab, arms in doc.items():
        if "ref" not in arms:
            continue
        r = np.array(arms["ref"]["ce"]); kind = np.array(arms["ref"]["kind"]); langs = np.array(arms["ref"]["langs"])
        for arm, v in arms.items():
            if arm == "ref":
                continue
            q = np.array(v["ce"]); s = "%-5s %-12s %7.1f МБ" % (lab, arm, v["bytes"] / 1e6)
            for name, m in (("текст", kind == "text"), ("en", langs == "en"), ("ru", langs == "ru"), ("sr", langs == "sr"), ("код", kind == "code")):
                d = (q - r)[m]; idx = rng.integers(0, len(d), (20000, len(d)))
                lo, hi = np.percentile(np.exp(d[idx].mean(1)) - 1, [2.5, 97.5])
                s += " | %s %+.2f%% [%+.2f; %+.2f]" % (name, 100 * (math.exp(d.mean()) - 1), 100 * lo, 100 * hi)
            print(s)


if __name__ == "__main__":
    report() if sys.argv[1] == "--report" else main()
