"""ppl-ГЕЙТ КВАНТОВАНИЯ LoRA-ВЕТОК ПОД mx.quantized_matmul.

ПОЧЕМУ ЭТОТ ЗАМЕР ОБЯЗАТЕЛЕН. Соблазнительно считать перевод LoRA в
контейнер mlx-affine РЕПАКОМ: в .rwkvq они лежат в asym gw64, то есть
w = q*scale + min по группам 64 -- ровно тот же контейнер. Но writer
квантует LoRA по СЫРЫМ ключам state_dict, ДО транспозиции, поэтому группы
там идут вдоль ВЫХОДНОЙ оси матмула, а quantized_matmul требует их вдоль
ВХОДНОЙ. Это ДРУГОЕ квантование: другая ось, другие группы, другие
значения. Ожидание хорошее (изолированный вклад веток при шести битах
+0.006% на 19418 предсказаниях, замер 12.08), но ожидание -- не замер.

ПАРНО И В ОДНОМ ПРОЦЕССЕ. Модель строится ОДИН раз, конфиги отличаются
ровно флагом qm.LORA_Q, корпус тот же -- поэтому разность считается по
ОДНИМ И ТЕМ ЖЕ последовательностям, и разброс самого корпуса из неё
сокращается. Закон 2 (один конфиг на процесс) писался про тяжёлые прогоны
с воркспейсом грид-поиска; здесь ничего не квантуется на лету, добавка --
десятки мегабайт квантованных буферов.

ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ОБЯЗАТЕЛЬНЫ: ожидаемый эффект (сотые доли
процента) лежит ниже полосы различимости корпуса (+-0.05-0.09 п.п.), и
без интервала «стало лучше на 0.02%» -- это шум, выданный за результат.
Бутстрэп ПАРНЫЙ, по последовательностям, 20000 итераций.

    python tests/eval_lora_quant.py [model.rwkvq] [nseq]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 38
SEQLEN = 512
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
OUT = os.environ.get("RWKVQ_LORA_OUT", "/tmp/lora_quant_ppl.json")
BOOT = 20000

CONFIGS = [("fp16", None, 0), ("sep@8", "sep", 8), ("sep@6", "sep", 6),
           ("glue@8", "glue", 8)]
_only = os.environ.get("RWKVQ_LORA_CONFIGS")     # напр. "fp16,sep@8"
if _only:
    keep = _only.split(",")
    CONFIGS = [c for c in CONFIGS if c[0] in keep]


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def nll_per_seq(model, data):
    """По одной последовательности: логиты [1, 511, 65536] в fp32 -- уже
    1.07 ГБ, батчить нечем, да и разбивка по языкам всё равно нужна
    попоследовательностная."""
    out = []
    for i in range(data.shape[0]):
        b = data[i:i + 1]
        logits = model(mx.array(b[:, :-1])).astype(mx.float32)
        tgt = mx.array(b[:, 1:])[..., None]
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, tgt, axis=-1).squeeze(-1)
        mx.eval(nll)
        a = np.array(nll)
        out.append((float(a.sum()), int(a.size)))
        del logits, nll
    return out


def ppl_of(pairs, idx=None):
    idx = range(len(pairs)) if idx is None else idx
    s = sum(pairs[i][0] for i in idx)
    n = sum(pairs[i][1] for i in idx)
    mean = s / n
    if not (mean > 1e-6) or mean != mean:       # закон 21
        raise RuntimeError(f"ppl вырождена (NLL={mean!r}): смотрите stderr")
    return float(np.exp(mean))


def boot_ci(base, cand, rng):
    """Парный бутстрэп по последовательностям: дельта в процентах."""
    n = len(base)
    d = np.empty(BOOT)
    idx = rng.integers(0, n, size=(BOOT, n))
    bs = np.array([x[0] for x in base]); bn = np.array([x[1] for x in base])
    cs = np.array([x[0] for x in cand]); cn = np.array([x[1] for x in cand])
    for i in range(BOOT):
        j = idx[i]
        pb = np.exp(bs[j].sum() / bn[j].sum())
        pc = np.exp(cs[j].sum() / cn[j].sum())
        d[i] = (pc / pb - 1) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    blob = torch.load(CORPUS)
    data = blob["tokens"][:NSEQ, :SEQLEN].numpy()
    langs = list(blob["lang"])[:NSEQ]
    print(f"корпус {os.path.basename(CORPUS)}: {tuple(data.shape)}, "
          f"{data.shape[0] * (SEQLEN - 1)} предсказаний, "
          f"языки {dict((l, langs.count(l)) for l in sorted(set(langs)))}")
    sw0 = swap_mb()
    model = qm.QuantRWKV7(load_raw(PATH))
    print(f"модель собрана, своп {sw0:.0f} -> {swap_mb():.0f} МБ", flush=True)

    res = {}
    # ppl считается ПРЕФИЛЛОМ, а по умолчанию квантованные ветки живут
    # только в декоде -- без этой строки замер сравнивал бы fp16 с fp16 и
    # показал бы «квантование ничего не стоит» на пустом месте (закон 15).
    qm.LORA_Q_DECODE_ONLY = False
    for name, mode, bits in CONFIGS:
        qm.LORA_Q, qm.LORA_QBITS = mode, bits
        qm.reset_lora_q(model)
        if mode:
            for b in model.blocks:
                b.tmix._build_lora_q()
            built = sum(b.tmix._lq_A is not None for b in model.blocks)
            assert built == len(model.blocks), f"{name}: буферы не построены"
        t0 = time.time()
        pairs = nll_per_seq(model, data)
        res[name] = pairs
        print(f"{name:>7}: ppl {ppl_of(pairs):.5f}  ({time.time()-t0:.0f} с, "
              f"своп {swap_mb():.0f} МБ)", flush=True)
    qm.LORA_Q = None

    rng = np.random.default_rng(20260815)
    base = res["fp16"]
    p0 = ppl_of(base)
    print(f"\nбаза (LoRA плотным fp16, то есть деквант .rwkvq как есть): "
          f"ppl {p0:.5f}")
    print(f"{'конфиг':>8} | {'ppl':>9} {'Δ%':>8} | {'95% CI':>20} | значимо |"
          f" " + "  ".join(f"{l:>7}" for l in sorted(set(langs))))
    for name, mode, bits in CONFIGS:
        if name == "fp16":
            continue
        p = ppl_of(res[name])
        lo, hi = boot_ci(base, res[name], rng)
        sig = "да" if (lo > 0 or hi < 0) else "НЕТ"
        by = []
        for l in sorted(set(langs)):
            ii = [i for i, x in enumerate(langs) if x == l]
            by.append((ppl_of(res[name], ii) / ppl_of(base, ii) - 1) * 100)
        print(f"{name:>8} | {p:9.5f} {(p/p0-1)*100:+7.3f}% | "
              f"[{lo:+7.3f}; {hi:+7.3f}] | {sig:>7} | "
              + "  ".join(f"{v:+6.3f}%" for v in by))

    json.dump({"path": PATH, "corpus": CORPUS, "nseq": NSEQ, "langs": langs,
               "nll": {k: v for k, v in res.items()}},
              open(OUT, "w"), ensure_ascii=False)
    print(f"\nper-sequence NLL -> {OUT} (интервалы пересчитываются без "
          f"повторного прогона)")
    print(f"своп: {sw0:.0f} -> {swap_mb():.0f} МБ")


if __name__ == "__main__":
    main()
