"""w/a/v_lora: 6 бит против 8 при ОДИНАКОВОМ размере файла.

Утверждение, которое проверяется. Контейнер `asym gw64`
(`_make_qt_gw_asym`) хранит коды в uint8, а scale/min -- в fp32 на блок
из 64, поэтому стоит 9.000 бит/вес при bits=5, 6 И 8 -- измерено,
tests/probe_schema_cost.py. Значит битность там не влияет на размер
вовсе и является ЧИСТЫМ параметром качества. В пресетах стоит 6.

Проверяется в два шага, и первый обязателен:

  РАЗМЕР. Оба конфига квантуются в настоящие файлы, и размеры
  сравниваются побайтово. Утверждение "не влияет на размер" выведено из
  одного тензора; на модели целиком его надо предъявить, а не
  экстраполировать (закон 17 про синтетику, распространённый на выводы).

  КАЧЕСТВО. ppl обоих конфигов на мультиязычном корпусе.

Если размеры совпали, а ppl улучшилось -- это выигрыш, за который не
заплачено ничем, и такие в проекте редки.

    python tests/ablate_lora_bits_free.py size      # только размеры
    python tests/ablate_lora_bits_free.py ppl 6     # ppl одного конфига
    python tests/ablate_lora_bits_free.py ppl 8
    python tests/ablate_lora_bits_free.py --report

Один конфиг на процесс для ppl (закон 13). act_stats выключены явно:
иначе результат зависит от содержимого /tmp.
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
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.presets import PRESETS

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
PRESET = os.environ.get("RWKVQ_PRESET", "reduction")
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 16))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))
BATCH = int(os.environ.get("RWKVQ_BATCH", 2))
OUT = os.environ.get("RWKVQ_OUT", f"/tmp/lora_bits_{PRESET}.json")
LORA = ("w_lora", "a_lora", "v_lora")


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"],
                          capture_output=True, text=True).stdout.strip()


def variant_iso(bits):
    """Квантованы ТОЛЬКО w/a/v_lora, всё остальное bf16.

    Изолированный замер здесь чувствительнее композитного, и это не
    придирка к методике: ожидаемый эффект мал (по замеру развилки
    QLoRA-базы вся группа стоит +0.024% на 1.5B), а в композите он тонет
    в деградации от proj/cmix/emb/head, которые на порядок больше.
    Композит всё равно нужен -- изолированный выигрыш не переносится
    (закон 5), -- но начинать надо с того замера, который вообще способен
    увидеть разницу.
    """
    return QuantConfig(group_scale={g: 64 for g in LORA},
                       group_scale_mode={g: "asym" for g in LORA},
                       act_stats_path=None,
                       **{g: bits for g in LORA})


def variant(bits):
    base = PRESETS[PRESET]
    c = QuantConfig(group_scale=dict(base.group_scale),
                    group_scale_mode=dict(base.group_scale_mode),
                    clip_percentiles=dict(base.clip_percentiles),
                    outlier_fracs=dict(base.outlier_fracs),
                    bits_overrides=dict(base.bits_overrides),
                    act_stats_path=None,
                    **dict(base.bits))
    for g in LORA:
        c.bits[g] = bits
    return c


def load_doc():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"ckpt": CKPT, "preset": PRESET}


def save_doc(d):
    with open(OUT, "w") as f:
        json.dump(d, f, indent=1)


def do_size():
    """Квантуем оба конфига в реальные файлы и сравниваем побайтово."""
    from rwkv_quant.api import quantize
    doc = load_doc()
    sizes = {}
    for bits in (6, 8):
        path = f"/tmp/lora{bits}_{PRESET}.rwkvq"
        t0 = time.time()
        quantize(CKPT, path, config=variant(bits), real_gw=True, verbose=False)
        sizes[str(bits)] = os.path.getsize(path)
        print(f"  lora={bits}: {sizes[str(bits)]:,} байт "
              f"({sizes[str(bits)]/1e6:.1f} МБ)  [{time.time()-t0:.0f}s]",
              flush=True)
    d = sizes["8"] - sizes["6"]
    doc["sizes"] = sizes
    save_doc(doc)
    print(f"\nразница: {d:+,} байт", end="")
    if d == 0:
        print("  -> РАЗМЕР ИДЕНТИЧЕН, битность там бесплатна")
    else:
        print(f"  -> НЕ бесплатно ({d/1e6:+.2f} МБ), утверждение неверно")
    # содержимое обязано различаться -- иначе мы сравниваем файл сам с собой
    import hashlib
    h = {}
    for bits in (6, 8):
        with open(f"/tmp/lora{bits}_{PRESET}.rwkvq", "rb") as f:
            hh = hashlib.sha256()
            while chunk := f.read(1 << 22):
                hh.update(chunk)
            h[bits] = hh.hexdigest()[:16]
    print(f"sha256: 6 -> {h[6]}, 8 -> {h[8]}  "
          f"{'(различаются, как и должны)' if h[6] != h[8] else '(ОДИНАКОВЫ -- конфиг не применился!)'}")
    assert h[6] != h[8], "файлы побитово одинаковы: битность не применилась"
    return d == 0


def do_ppl(which, iso=False):
    print(f"своп до: {swap()}", flush=True)
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")
    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    doc = load_doc()
    n_pred = int(data.shape[0] * (data.shape[1] - 1))
    doc["n_pred"] = n_pred
    t0 = time.time()
    if which == "bf16":
        p = perplexity(model, data, QuantConfig(), batch_size=BATCH)
        doc["bf16_ppl"] = p
        print(f"bf16 ppl={p:.4f}  [{time.time()-t0:.0f}s]")
    else:
        bits = int(which)
        cfg = variant_iso(bits) if iso else variant(bits)
        p = perplexity(model, data, cfg, batch_size=BATCH)
        doc.setdefault("ppl_iso" if iso else "ppl", {})[str(bits)] = p
        bf = doc.get("bf16_ppl")
        dd = f"{100*(p-bf)/bf:+.3f}%" if bf else "(bf16 ещё не мерен)"
        print(f"lora={bits} ppl={p:.4f} Δ={dd}  [{time.time()-t0:.0f}s]")
    save_doc(doc)
    report(doc)
    print(f"своп после: {swap()}")


def report(doc):
    bf = doc.get("bf16_ppl")
    sizes = doc.get("sizes", {})
    for tag, key in (("ИЗОЛИРОВАННО (только w/a/v_lora)", "ppl_iso"),
                     ("КОМПОЗИТ (пресет целиком)", "ppl")):
        if doc.get(key):
            print(f"\n{tag}")
            _one_table(doc, doc[key], bf, sizes)
    return


def _one_table(doc, ppl, bf, sizes):
    print(f"\n{'lora бит':>9s} {'ppl':>10s} {'Δ к bf16':>10s} {'байт':>14s}")
    for b in ("6", "8"):
        if b not in ppl and b not in sizes:
            continue
        p = ppl.get(b)
        dd = f"{100*(p-bf)/bf:+.3f}%" if (p and bf) else "--"
        print(f"{b:>9s} {p if p else float('nan'):10.4f} {dd:>10s} "
              f"{sizes.get(b, 0):14,}")
    if "6" in ppl and "8" in ppl:
        gain = (ppl["6"] - ppl["8"]) / ppl["6"] * 100
        same = sizes.get("6") == sizes.get("8") if sizes else None
        d6 = 100 * (ppl["6"] - bf) / bf if bf else None
        print(f"\n8 бит против 6: ppl {'лучше' if gain > 0 else 'ХУЖЕ'} на "
              f"{abs(gain):.3f}%"
              + (", размер идентичен" if same
                 else ", размер не сверен" if same is None else ", размер вырос"))
        # Формулировка «выигрыш даром» здесь была бы подтасовкой. Если сама
        # группа при шести битах стоит околонуля, то у восьми нет запаса,
        # который можно отыграть, и любая разница -- шум прогона.
        if d6 is not None and abs(d6) < 0.05:
            print(f"ОСТОРОЖНО: группа при 6 битах стоит всего {d6:+.3f}% -- "
                  f"это уровень шума, и разница между 6 и 8 битами по ppl\n"
                  f"НЕ ИЗМЕРИМА в принципе. Довод в пользу восьми остаётся "
                  f"только структурный:\nразмер тот же, сетка вчетверо мельче, "
                  f"хуже стать не может.")


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--report":
        report(load_doc())
        print("\nИспользование: ablate_lora_bits_free.py size | iso <bf16|6|8> | ppl <bf16|6|8>")
        return
    if a[0] == "size":
        do_size()
    elif a[0] == "ppl":
        do_ppl(a[1])
    elif a[0] == "iso":
        do_ppl(a[1], iso=True)


if __name__ == "__main__":
    main()
