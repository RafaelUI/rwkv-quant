"""AFFINE КАК РАСКЛАДКА В ПАМЯТИ ПРИ НЕИЗМЕННОМ ФАЙЛЕ: качество.

ЗАМЫСЕЛ. На префилле мы уходим с кернеля на `_dequant_w` + плотный матмул
и проигрываем нативному `mx.quantized_matmul` 2.2-2.5x на слое (разведка
`probe_prefill_affine`). Держать affine ВТОРОЙ раскладкой рядом с нашей
нельзя -- это +1.5 ГБ на 1.5B. Но её можно держать ЕДИНСТВЕННОЙ В ПАМЯТИ,
оставив файл как есть: на диске сохраняется наша сетка (sym + AW, ради
которой всё и делалось), а в RAM лежит её affine-транскрипция.

ТОЧНОГО репака sym НЕ СУЩЕСТВУЕТ: блок у sym -- 16, а MLX принимает
group_size только 32/64/128 (проверено, не выведено). Значит это ПЕРЕ-
квантование поверх нашего, то есть двойное, и ppl обязателен.

Два кандидата, и они различаются знаком по памяти:
  affine@6 gs=64 -- 6.500 бит/вес против наших 6.5625: память и трафик
                    декода НЕ РАСТУТ вовсе;
  affine@8 gs=64 -- 8.500 бит/вес: +30% резидентно и +30% трафика декода,
                    зато значения вчетверо ближе к нашим (rel 5e-3
                    против 2.3e-2 на выходе слоя).

Один конфиг на процесс (закон 2), пары сходятся по одному и тому же
корпусу; интервалы -- парным бутстрэпом в --report.

    python tests/eval_affine_inmem.py base|a6|a8
    python tests/eval_affine_inmem.py --report
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

PATH = os.environ.get("RWKVQ_MODEL", "/tmp/reduction_sym_head8.rwkvq")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
NSEQ, SEQLEN = 38, 512
OUT = os.environ.get("RWKVQ_AFFINE_OUT", "/tmp/affine_inmem_ppl.json")
BOOT = 20000
MODES = {"base": None, "a6": (6, 64), "a8": (8, 64)}


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


class AffineLinear:
    """Тот же интерфейс __call__(x) -> y, что у Sym/GwQuantLinear, но
    поверх нативного контейнера MLX. Отдельный класс, а не ветка в
    MlxAffineQuantLinear: тот читает поля чужого чекпоинта (qt.mlx_*), а
    здесь буферы получены пересчётом из НАШЕЙ раскладки."""

    def __init__(self, w, gs, bits):
        self.wq, self.scales, self.biases = mx.quantize(w, group_size=gs,
                                                        bits=bits)
        self.gs, self.bits = gs, bits
        self.out_features, self.in_features = int(w.shape[0]), int(w.shape[1])

    def __call__(self, x):
        y = mx.quantized_matmul(x.astype(mx.float16), self.wq,
                                scales=self.scales, biases=self.biases,
                                transpose=True, group_size=self.gs,
                                bits=self.bits)
        return y.astype(x.dtype)


def to_affine(model, bits, gs):
    """Перевести все матричные слои в affine ПО ОДНОМУ, немедленно
    отпуская исходный: держать обе раскладки разом -- это +1.5 ГБ, то
    есть ровно то, чего замысел избегает."""
    n = 0
    slots = [(model, "head")]
    for b in model.blocks:
        slots += [(b.tmix, "r_proj"), (b.tmix, "k_proj"), (b.tmix, "v_proj"),
                  (b.tmix, "o_proj"), (b.cmix, "key"), (b.cmix, "value")]
    for obj, attr in slots:
        lin = getattr(obj, attr)
        w = lin._dequant_w() if hasattr(lin, "_dequant_w") else lin.w
        mx.eval(w)
        setattr(obj, attr, AffineLinear(w, gs, bits))
        del lin, w
        mx.eval([getattr(obj, attr).wq])
        # чистить кеш аллокатора НА КАЖДОМ слое, а не в конце: иначе
        # освобождённые буферы нашей раскладки висят в нём до конца
        # перевода, и пик описывает две модели вместо одной (на этом
        # замер с двумя моделями в одном процессе вырос до 7 ГБ)
        mx.clear_cache()
        n += 1
    return n


def live_mb(model):
    seen, tot = set(), 0

    def walk(o, d=0):
        nonlocal tot
        if d > 4:
            return
        for v in vars(o).values():
            for it in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(it, mx.array):
                    if id(it) not in seen:
                        seen.add(id(it))
                        tot += it.nbytes
                elif hasattr(it, "__dict__"):
                    walk(it, d + 1)
    walk(model)
    return tot / 1e6


def nll_per_seq(model, data):
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
    s = sum(pairs[i][0] for i in idx); n = sum(pairs[i][1] for i in idx)
    mean = s / n
    if not (mean > 1e-6) or mean != mean:
        raise RuntimeError(f"ppl вырождена (NLL={mean!r})")
    return float(np.exp(mean))


def report():
    doc = json.load(open(OUT))
    rows = doc["rows"]
    langs = doc["langs"]
    base = [tuple(x) for x in rows["base"]["nll"]]
    p0 = ppl_of(base)
    rng = np.random.default_rng(20260816)
    print(f"база (наша раскладка sym, точный деквант): ppl {p0:.5f}, "
          f"живые буферы {rows['base']['live_mb']:.1f} МБ")
    print(f"{'конфиг':>6} | {'ppl':>9} {'Δ%':>8} | {'95% CI':>20} | знач. | "
          f"{'живые МБ':>9} | " + "  ".join(f"{l:>7}" for l in sorted(set(langs))))
    for k in ("a6", "a8"):
        if k not in rows:
            continue
        cand = [tuple(x) for x in rows[k]["nll"]]
        p = ppl_of(cand)
        bs = np.array([x[0] for x in base]); bn = np.array([x[1] for x in base])
        cs = np.array([x[0] for x in cand]); cn = np.array([x[1] for x in cand])
        idx = rng.integers(0, len(base), size=(BOOT, len(base)))
        d = np.array([(np.exp(cs[j].sum() / cn[j].sum())
                       / np.exp(bs[j].sum() / bn[j].sum()) - 1) * 100
                      for j in idx])
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        by = []
        for l in sorted(set(langs)):
            ii = [i for i, x in enumerate(langs) if x == l]
            by.append((ppl_of(cand, ii) / ppl_of(base, ii) - 1) * 100)
        print(f"{k:>6} | {p:9.5f} {(p/p0-1)*100:+7.3f}% | "
              f"[{lo:+7.3f}; {hi:+7.3f}] | {'да' if (lo>0 or hi<0) else 'НЕТ':>5} | "
              f"{rows[k]['live_mb']:9.1f} | "
              + "  ".join(f"{v:+6.3f}%" for v in by))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "--report":
        return report()
    blob = torch.load(CORPUS)
    data = blob["tokens"][:NSEQ, :SEQLEN].numpy()
    langs = list(blob["lang"])[:NSEQ]
    sw0 = swap_mb()
    model = qm.QuantRWKV7(load_raw(PATH))
    if MODES[mode]:
        bits, gs = MODES[mode]
        t0 = time.time()
        n = to_affine(model, bits, gs)
        print(f"переведено слоёв {n} в affine@{bits} gs={gs} за "
              f"{time.time()-t0:.1f} с", flush=True)
    lm = live_mb(model)
    print(f"живые буферы {lm:.1f} МБ, своп {sw0:.0f} -> {swap_mb():.0f}",
          flush=True)
    t0 = time.time()
    pairs = nll_per_seq(model, data)
    print(f"{mode}: ppl {ppl_of(pairs):.5f} ({time.time()-t0:.0f} с, "
          f"своп {swap_mb():.0f} МБ)", flush=True)
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    doc["langs"] = langs
    doc["path"] = PATH
    doc["rows"][mode] = {"nll": pairs, "live_mb": lm}
    json.dump(doc, open(OUT, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
