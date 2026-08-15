"""emb и head порознь: сколько стоит их квантование и чем это лечится.

ЗАЧЕМ. После композита 13.08 `emb`/`head` -- последняя большая группа, где
Q6_K-раскладка не пробована ВООБЩЕ, и единственная, про которую в проекте
есть только одно изолированное число (10.08, asym_sb6): при шести битах
emb +0.01%, head +0.16%, то есть head в 16 раз чувствительнее. Вместе они
29% модели, так что даже +0.16% -- это не мелочь, а треть нынешнего
бюджета REDUCTION на 1.5B.

ЧТО МЕРИТСЯ. Изолированно: квантуется РОВНО одна группа, всё прочее
bf16. Три рычага, все в одном прогоне:
  1. РАСКЛАДКА -- `sym_aw` (Q6_K, блок 16) против нынешнего
     `asym_sb6_aw` (блок 32). На cmix это дало -0.371/-0.463 п.п., на
     proj -0.033/-0.139. Цена +0.0625 бита/вес.
  2. БИТНОСТЬ -- `asym` gw64 при 8 битах. У sb6 нет 7 и 8 бит (упаковщик
     умеет 4/5/6, кернель знает xbits 0/1/2), поэтому дать группе больше
     шести бит можно ТОЛЬКО уходом в asym gw64 за 9.000 бит/вес. На head
     это +42 МБ на 1.5B -- в рамке REDUCTION («+50 МБ ради качества
     приемлемы») ровно по бюджету.
  3. КОНТРОЛЬ БИТНОСТИ -- тот же asym gw64 при 6 битах. Контейнер стоит
     9.000 бит/вес при 5, 6 и 8 одинаково (probe_schema_cost), то есть
     размер тут НЕ зависит от битности, и разница 6 против 8 -- чистая
     цена сетки. Без этой пары нельзя сказать, что дало эффект:
     контейнер или два лишних бита.

ПОРОГ ШУМА, КОТОРОГО В ПРОЕКТЕ НЕ БЫЛО. До сих пор «в пределах шума»
говорилось на глаз. Здесь у каждой дельты есть доверительный интервал:
парный бутстрэп ПО ПОСЛЕДОВАТЕЛЬНОСТЯМ корпуса (те же индексы для
конфига и для bf16, потому что прогоны идут по одному и тому же тексту --
непарный бутстрэп раздул бы интервал вдвое и объявил шумом всё подряд).
Для этого в JSON сохраняются per-sequence NLL, а не только итоговая ppl:
интервалы потом пересчитываются без единого повторного прогона.

Читать так: если ноль лежит внутри интервала, разница НЕ измерена -- это
не «эффекта нет», а «этим корпусом не различить».

    python tests/ablate_emb_head.py bf16
    python tests/ablate_emb_head.py head_sym6
    python tests/ablate_emb_head.py --report
    python tests/ablate_emb_head.py --report /tmp/emb_head_2p9b.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.calibration.outlier_scan import GROUP_KEY_PATTERNS  # noqa: E402
from rwkv_quant.models.rwkv7_ref import RWKV7Ref  # noqa: E402

# предквантование и его гейт живут в композитном драйвере -- одна
# реализация на оба замера, чтобы «мерим одно, деплоим другое» не
# завелось ещё и здесь
from ablate_sym_composite import (bits_per_weight, mem_line,  # noqa: E402
                                  nll_by_seq, prequantize)

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
ACT = os.environ.get("RWKVQ_ACT_STATS", "/tmp/act_stats_1p5b_ml.pt")
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 38))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))
BATCH = int(os.environ.get("RWKVQ_BATCH", 2))
OUT = os.environ.get("RWKVQ_OUT", "/tmp/emb_head_1p5b.json")
NBOOT = int(os.environ.get("RWKVQ_NBOOT", 20000))


def iso(group, bits, gs, mode):
    """Конфиг, где квантована РОВНО одна группа."""
    return QuantConfig(**{group: bits}, group_scale={group: gs},
                       group_scale_mode={group: mode}, act_stats_path=ACT)


# имя -> (группа, биты, блок, режим, что проверяем)
CONFIGS = {
    "bf16": (None, 0, 0, None, "якорь"),
    "emb_asym6":    ("emb", 6, 32, "asym_sb6_aw", "как в пресете"),
    "emb_sym6":     ("emb", 6, 16, "sym_aw", "Q6_K-раскладка"),
    "emb_gw64_6":   ("emb", 6, 64, "asym", "контейнер gw64, 6 бит"),
    "emb_gw64_8":   ("emb", 8, 64, "asym", "тот же контейнер, 8 бит"),
    "head_asym6":   ("head", 6, 32, "asym_sb6_aw", "как в пресете"),
    "head_sym6":    ("head", 6, 16, "sym_aw", "Q6_K-раскладка"),
    "head_gw64_6":  ("head", 6, 64, "asym", "контейнер gw64, 6 бит"),
    "head_gw64_8":  ("head", 8, 64, "asym", "тот же контейнер, 8 бит"),
    # ДОБАВЛЕНО ПОСЛЕ ПЕРВОГО ПРОГОНА на 1.5B. Он показал, что на head
    # работают ОБА рычага и они не конкурируют: раскладка даёт -0.066 п.п.
    # за 1 МБ, восемь бит в gw64 -- -0.096 п.п. за 42. Напрашивается
    # третья точка, которой в наборе не было: восемь бит В Q6_K-РАСКЛАДКЕ.
    # Она стоит 8 + 8/16 + 16/256 = 8.5625 бит/вес, то есть ДЕШЕВЛЕ gw64
    # (9.000) при вдвое меньшем блоке и 8-битном scale. И упаковщику она
    # ничего не усложняет -- наоборот: при восьми битах коды это просто
    # байты, без битплоскостей, ради которых у sb6 заведены qh и qh2.
    "emb_sym8":     ("emb", 8, 16, "sym_aw", "Q6_K-раскладка, 8 бит"),
    "head_sym8":    ("head", 8, 16, "sym_aw", "Q6_K-раскладка, 8 бит"),
}


def cfg_of(name):
    g, bits, gs, mode, _ = CONFIGS[name]
    return QuantConfig() if g is None else iso(g, bits, gs, mode)


def group_mb(sd, name):
    g, bits, gs, mode, _ = CONFIGS[name]
    if g is None:
        return 0.0
    n = sum(t.numel() for k, t in sd.items()
            if t.dim() == 2 and any(k.endswith(p) or p in k
                                    for p in GROUP_KEY_PATTERNS[g]))
    return n * bits_per_weight(g, cfg_of(name)) / 8 / 1e6


# ------------------------------------------------------------- бутстрэп

def ppl_of(pairs, idx=None):
    idx = range(len(pairs)) if idx is None else idx
    s = sum(pairs[i][0] for i in idx)
    n = sum(pairs[i][1] for i in idx)
    return float(torch.exp(torch.tensor(s / n)))


def boot_ci(base_pairs, cfg_pairs, nboot=NBOOT, seed=0):
    """95% CI на Δ% ПАРНЫМ бутстрэпом по последовательностям.

    Парным -- потому что оба конфига прогнаны по ОДНОМУ тексту: на каждой
    итерации берётся одна и та же выборка индексов для base и cfg, и
    разброс самого корпуса (одни чанки просто труднее других) сокращается.
    Непарный бутстрэп мерил бы дисперсию корпуса, а не дисперсию эффекта,
    и объявил бы шумом почти всё.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(base_pairs)
    out = []
    for _ in range(nboot):
        idx = torch.randint(0, n, (n,), generator=g).tolist()
        b, c = ppl_of(base_pairs, idx), ppl_of(cfg_pairs, idx)
        out.append(100 * (c - b) / b)
    out.sort()
    lo = out[int(0.025 * nboot)]
    hi = out[int(0.975 * nboot)]
    return lo, hi


# --------------------------------------------------------------- отчёт

def load_doc(path=None):
    p = path or OUT
    return json.load(open(p)) if os.path.exists(p) else {"ckpt": CKPT,
                                                         "rows": {}}


def report(doc):
    rows = doc.get("rows", {})
    base = rows.get("bf16")
    if not base:
        print("нет якоря bf16")
        return
    bp = [tuple(x) for x in base["per_seq"]]
    bf = ppl_of(bp)
    langs = doc.get("langs", [])
    uniq = sorted(set(langs))
    print(f"\nизолированно, {doc.get('n_pred','?')} предсказаний, "
          f"{os.path.basename(doc.get('ckpt',''))}, bf16 ppl={bf:.5f}")
    print(f"бутстрэп: {NBOOT} итераций, парный, по {len(bp)} "
          f"последовательностям\n")
    print(f"{'конфиг':14s} {'бит/вес':>7s} {'МБ':>7s} {'ppl':>9s} "
          f"{'Δ':>8s} {'95% CI':>18s}  значимо  комментарий")
    for name, (g, bits, gs, mode, note) in CONFIGS.items():
        r = rows.get(name)
        if not r or name == "bf16":
            continue
        cp = [tuple(x) for x in r["per_seq"]]
        p = ppl_of(cp)
        d = 100 * (p - bf) / bf
        lo, hi = boot_ci(bp, cp)
        sig = "да" if (lo > 0 or hi < 0) else "НЕТ"
        bpw = bits_per_weight(g, cfg_of(name))
        print(f"{name:14s} {bpw:7.4f} {r.get('mb',0):7.1f} {p:9.4f} "
              f"{d:+7.3f}% [{lo:+6.3f}; {hi:+6.3f}]  {sig:>6s}  {note}")
    # попарные сравнения внутри группы: раскладка и битность против
    # нынешней схемы -- то, ради чего замер и ставился
    print()
    for g in ("emb", "head"):
        a = rows.get(f"{g}_asym6")
        if not a:
            continue
        ap = [tuple(x) for x in a["per_seq"]]
        for other, what in ((f"{g}_sym6", "Q6_K-раскладка"),
                            (f"{g}_gw64_8", "asym gw64 @8"),
                            (f"{g}_sym8", "Q6_K @8")):
            r = rows.get(other)
            if not r:
                continue
            cp = [tuple(x) for x in r["per_seq"]]
            d = 100 * (ppl_of(cp) - ppl_of(ap)) / bf
            lo, hi = boot_ci(ap, cp)
            sig = "значимо" if (lo > 0 or hi < 0) else "в пределах шума"
            print(f"{g}: {what:16s} против нынешней схемы: {d:+.3f} п.п. "
                  f"[{lo:+.3f}; {hi:+.3f}] {sig}, "
                  f"{r.get('mb',0)-a.get('mb',0):+.1f} МБ")
    if uniq:
        print(f"\nпо языкам (Δ% к bf16):")
        print(f"{'конфиг':14s}" + "".join(f" {l:>9s}" for l in uniq))
        for name in CONFIGS:
            r = rows.get(name)
            if not r or name == "bf16":
                continue
            line = f"{name:14s}"
            for l in uniq:
                sel = [i for i, ll in enumerate(langs) if ll == l]
                pl, bl = ppl_of([tuple(x) for x in r["per_seq"]], sel), \
                    ppl_of(bp, sel)
                line += f" {100*(pl-bl)/bl:+8.3f}%"
            print(line)


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--report":
        report(load_doc(a[1] if len(a) > 1 else None))
        print(f"\nконфиги: {' '.join(CONFIGS)}")
        return 0
    name = a[0]
    if name not in CONFIGS:
        raise SystemExit(f"неизвестный конфиг {name}; есть: {list(CONFIGS)}")
    if name != "bf16" and not os.path.exists(ACT):
        raise SystemExit(f"нет статистики активаций {ACT}: AW-режим молча "
                         f"выродится в _search (закон 15)")

    print(f"=== {os.path.basename(CKPT)} / {name} ===", flush=True)
    print(f"старт: {mem_line()}", flush=True)
    blob = torch.load(CORPUS)
    tok = blob["tokens"] if isinstance(blob, dict) else blob
    langs = list(blob.get("lang", []))[:NSEQ] if isinstance(blob, dict) else []
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")

    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    t0 = time.time()
    if name != "bf16":
        prequantize(model, cfg_of(name), verbose=False)
        print(f"после кванта: {mem_line()}", flush=True)
    pairs = nll_by_seq(model, data, batch=BATCH)

    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    doc = load_doc()
    doc["langs"] = langs
    doc["n_pred"] = sum(x[1] for x in pairs)
    doc["rows"][name] = {"per_seq": pairs, "mb": group_mb(sd, name),
                         "seconds": round(time.time() - t0)}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"{name}: ppl={ppl_of(pairs):.4f}  [{time.time()-t0:.0f}s]",
          flush=True)
    print(f"финиш: {mem_line()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
