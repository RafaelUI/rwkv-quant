"""Q6_K-подобная симметричная раскладка против нашей асимметричной, на cmix.

Открытый вопрос 2 из NEXT_SESSION, закрываемый замером на реальной
модели вместо гауссовой синтетики.

ПОЧЕМУ ИМЕННО cmix. По leave-one-out от 04.08 бюджет REDUCTION после
`small=16` определяют cmix (+0.33 п.п.), proj и emb_head (+0.18 каждый).
Битность там уже выбрана и снижать её нельзя, значит качество при том же
размере можно взять только сменой РАСКЛАДКИ. cmix -- крупнейший вклад,
с него и начинаем.

ЧТО СРАВНИВАЕТСЯ. Наш `asym_sb6`: блок 32, суперблок 8, асимметрия
(scale И min на блок, оба по 6 бит против пары fp16 на суперблок) --
6.500 бит/вес. Q6_K-подобный `sym`: блок 16, суперблок 16, БЕЗ min,
8-битный scale против одной fp16 на суперблок -- 6.5625 бит/вес.

Гипотеза (из докстринга groupwise_sym_fake_dequant): на шести битах
отдельный min не окупается -- распределение весов почти симметрично, а
платим мы за него дважды (6 бит на min И урезанный до 6 бит scale),
тогда как Q6_K тратит тот же бюджет на вдвое меньший блок с 8-битным
scale. На гауссовых весах при равном бюджете sym16 выигрывал 21% на
шести битах. На реальной модели НЕ МЕРЕНО ни разу.

Разница в размере +0.0625 бита/вес -- около +10 МБ на 1.5B. В рамке
REDUCTION («максимум качества, борьба за мегабайт -- задача
COMPRESSION») это приемлемая цена, если качество отвечает.

ВАЖНО: `sym` существует только в fake-пути. Реального упаковщика и
кернеля для него НЕТ (`real_gw=True` поднимет NotImplementedError). Этот
замер отвечает на вопрос «стоит ли их писать», и ни на какой другой.

Изолированно: квантован ТОЛЬКО cmix, всё прочее bf16. Композит после,
если изолированный ответ положительный (закон 5).

    python tests/ablate_sym_cmix.py bf16
    python tests/ablate_sym_cmix.py asym_sb6_aw
    python tests/ablate_sym_cmix.py sym_aw
    python tests/ablate_sym_cmix.py --report
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.calibration.ablation import perplexity
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.outlier_scan import GROUP_KEY_PATTERNS
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
ACT = os.environ.get("RWKVQ_ACT_STATS", "/tmp/act_stats_1p5b_ml.pt")
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 38))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))
BATCH = int(os.environ.get("RWKVQ_BATCH", 2))
GROUP = os.environ.get("RWKVQ_GROUP", "cmix")
BITS = int(os.environ.get("RWKVQ_BITS", 6))
OUT = os.environ.get("RWKVQ_OUT", f"/tmp/sym_{GROUP}.json")

# режим -> (размер блока, бит/вес). Числа проверяются ниже по формуле,
# а не переписаны на глаз.
MODES = {
    "asym_sb6":        (32, "наш, без поиска (proj в REDUCTION)"),
    "asym_sb6_search": (32, "наш + грид-поиск scale/min"),
    "asym_sb6_aw":     (32, "наш + поиск, взвешенный E[x^2] (cmix в REDUCTION)"),
    "sym_plain":       (16, "Q6_K-подобный, без поиска"),
    "sym":             (16, "Q6_K-подобный + поиск"),
    "sym_aw":          (16, "Q6_K-подобный + поиск, взвешенный E[x^2]"),
}


def bits_per_weight(mode, bits):
    gs = MODES[mode][0]
    if mode.startswith("sym"):          # scale int8 на блок + fp16 d на суперблок
        sb = max(1, 256 // gs)
        return bits + 8 / gs + 16 / (gs * sb)
    return bits + (6 + 6) / gs + (16 + 16) / (gs * 8)   # sb6: qs/qm + d/dm


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"],
                          capture_output=True, text=True).stdout.strip()


def cfg_for(mode):
    act = ACT if mode.endswith("_aw") else None
    if mode.endswith("_aw") and not os.path.exists(ACT):
        raise SystemExit(
            f"режим {mode} требует act_stats, а {ACT} нет. Собрать:\n"
            f"  python tests/collect_act_stats.py "
            f"~/Develop/WKV-kvant/act_calib_multiling.pt {ACT} ':'")
    return QuantConfig(**{GROUP: BITS},
                       group_scale={GROUP: MODES[mode][0]},
                       group_scale_mode={GROUP: mode},
                       act_stats_path=act)


def group_mb(sd, mode):
    n = sum(t.numel() for k, t in sd.items()
            if t.dim() == 2 and any(k.endswith(p) or p in k
                                    for p in GROUP_KEY_PATTERNS[GROUP]))
    return n * bits_per_weight(mode, BITS) / 8 / 1e6, n


def load_doc():
    return json.load(open(OUT)) if os.path.exists(OUT) else {
        "ckpt": CKPT, "group": GROUP, "bits": BITS, "rows": {}}


def report(doc):
    bf = doc.get("bf16_ppl")
    rows = doc.get("rows", {})
    if not rows:
        print("нет данных")
        return
    print(f"\nгруппа {doc['group']} @ {doc['bits']} бит, изолированно, "
          f"{doc.get('n_pred','?')} предсказаний, bf16 ppl={bf}")
    print(f"\n{'режим':18s} {'блок':>5s} {'бит/вес':>8s} {'ppl':>10s} "
          f"{'Δ к bf16':>9s} {'МБ группы':>10s}")
    base = rows.get("asym_sb6_aw") or rows.get("asym_sb6_search") \
        or rows.get("asym_sb6")
    for m in MODES:
        if m not in rows:
            continue
        r = rows[m]
        d = 100 * (r["ppl"] - bf) / bf if bf else float("nan")
        print(f"{m:18s} {MODES[m][0]:5d} {bits_per_weight(m, doc['bits']):8.4f} "
              f"{r['ppl']:10.4f} {d:+8.3f}% {r['mb']:10.1f}")
    if base and rows.get("sym_aw"):
        db = 100 * (base["ppl"] - bf) / bf
        ds = 100 * (rows["sym_aw"]["ppl"] - bf) / bf
        dmb = rows["sym_aw"]["mb"] - base["mb"]
        print(f"\nsym_aw против нашего лучшего асимметричного: "
              f"{ds - db:+.3f} п.п. при {dmb:+.1f} МБ")
        print("  -> " + ("Q6_K-раскладка ЛУЧШЕ, писать упаковщик и кернель"
                         if ds < db else
                         "Q6_K-раскладка НЕ лучше на этой модели; на гауссе "
                         "выигрывала -- значит выигрыш не переносится"))


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--report":
        report(load_doc())
        print(f"\nрежимы: {' '.join(MODES)}")
        return
    which = a[0]
    print(f"своп до: {swap()}", flush=True)
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")
    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    doc = load_doc()
    doc["n_pred"] = int(data.shape[0] * (data.shape[1] - 1))
    t0 = time.time()
    if which == "bf16":
        p = perplexity(model, data, QuantConfig(), batch_size=BATCH)
        doc["bf16_ppl"] = p
        print(f"bf16 ppl={p:.4f}  [{time.time()-t0:.0f}s]")
    else:
        if which not in MODES:
            raise SystemExit(f"неизвестный режим {which}; есть: {list(MODES)}")
        p = perplexity(model, data, cfg_for(which), batch_size=BATCH)
        mb, n = group_mb(sd, which)
        doc["rows"][which] = {"ppl": p, "mb": mb, "n_params": n}
        bf = doc.get("bf16_ppl")
        dd = f"{100*(p-bf)/bf:+.3f}%" if bf else "(bf16 не мерен)"
        print(f"{which}: ppl={p:.4f} Δ={dd}  {mb:.1f} МБ "
              f"({bits_per_weight(which, BITS):.4f} бит/вес)  "
              f"[{time.time()-t0:.0f}s]")
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    report(doc)
    print(f"своп после: {swap()}")


if __name__ == "__main__":
    main()
