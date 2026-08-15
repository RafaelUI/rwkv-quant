"""Разложение групп на составляющие: ГДЕ ИМЕННО внутри proj/cmix/emb/head
рождается деградация.

ПОЧЕМУ НЕ ppl. Leave-one-out по группам (ablate_group_contrib, 04.08)
нашёл `small` -- 147K параметров, 0.01% модели, две трети деградации.
Он работал, потому что бюджет был +2.36%. Сейчас бюджет +0.111% на
fake-пути, а полоса различимости на этом корпусе ±0.05-0.09 п.п.
(парный бутстрэп, 19418 предсказаний). Разложить +0.111% на восемь частей
по ppl НЕВОЗМОЖНО В ПРИНЦИПЕ: каждая часть около 0.01 п.п., впятеро ниже
шума, сколько прогонов ни делай.

ЧЕМ МЕРИМ ВМЕСТО НЕЁ.

1. **KL(bf16 ‖ квантованная) на токен.** ppl смотрит вероятность ОДНОГО
   правильного токена; KL сравнивает распределения целиком -- 65536
   логитов на каждый токен. Сигнала на четыре порядка больше при том же
   прогоне, и эффект виден там, где Δppl тонет. Эталон (логиты bf16)
   считается один раз и кладётся на диск в float32: fp16 тут не годится
   -- при логитах порядка 10 её шаг сравним с самим измеряемым
   эффектом, то есть инструмент оказался бы грубее ppl, ради ухода от
   которой он и делается.

2. **KL ПО ПОЗИЦИИ В ПОСЛЕДОВАТЕЛЬНОСТИ.** Это специфично для RWKV: у
   трансформера ошибка на токене t не влияет на t+1 (KV-кэш пересчитан
   из тех же весов), у RWKV она входит в состояние и копится -- отсюда
   измеренный рост деградации с контекстом (+1.52% на 128 токенах против
   +2.57% на 512). Плоский профиль по t означает локальную ошибку
   считывания, растущий -- ошибку, вошедшую в рекуррентность. Это
   разные болезни и лечатся они по-разному.

3. **Расхождение residual stream по слоям** (`--trace`): ‖Δx_l(t)‖/‖x_l(t)‖
   у bf16 и квантованной модели. Отвечает на вопрос «в каком слое
   родилась ошибка» -- по ppl и даже по KL этого не видно, там только
   итог.

ПОДМНОЖЕСТВА выбираются регуляркой по ключу, и это те швы, где
«мелочь портит всё» правдоподобна структурно:

  proj_v / proj_k  -- пишут в состояние, ошибка копится по длине;
  proj_r / proj_o  -- только читают текущий шаг, ошибка живёт один токен;
  vfirst           -- blocks.0.att.value.weight. На нулевом слое
                      `v_first = v`, и КАЖДЫЙ последующий слой
                      подмешивает этот v_first (см. _tmix_forward).
                      Одна матрица из 96 в группе proj, 1% группы, с
                      влиянием на весь стек -- ровно профиль `small`;
  layer0           -- весь нулевой блок: его выход входит в состояние на
                      всех последующих шагах;
  cmix_key         -- выход идёт в relu(·)^2, то есть ошибка в строках с
                      отрицательной пре-активацией зануляется точно;
  cmix_value       -- вход неотрицателен и сильно скошен.

Режимы: --only (квантуется ТОЛЬКО подмножество, изолированно) и
--except (квантуется всё, КРОМЕ него -- leave-one-out). Конфиг берётся
из ablate_sym_composite, а не переписывается: разъехавшиеся списки
превратили бы разложение в сравнение разных схем.

Один конфиг на процесс (законы 11 и 13). Память -- системными командами.

    python tests/ablate_subgroups.py --ref
    python tests/ablate_subgroups.py reduction_sym_head8
    python tests/ablate_subgroups.py reduction_sym_head8 --except vfirst
    python tests/ablate_subgroups.py reduction_sym_head8 --only proj_v --trace
    python tests/ablate_subgroups.py --report
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.calibration import fake_quant  # noqa: E402
from rwkv_quant.models.rwkv7_ref import RWKV7Ref  # noqa: E402

import ablate_sym_composite as comp  # noqa: E402

CKPT, CORPUS = comp.CKPT, comp.CORPUS
NSEQ = int(os.environ.get("RWKVQ_KL_NSEQ", 8))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))
# Точки в имени чекпоинта значащие: "rwkv7-g1h-2.9b" и "rwkv7-g1h-1.5b"
# после split(".")[0] превращаются в "rwkv7-g1h-2" и "rwkv7-g1h-1" --
# работает, но ровно до первой пары имён, различающихся после первой
# точки. Файл с именем одного чекпоинта и содержимым другого в этом
# проекте уже случался (закон 15), поэтому имя берётся целиком.
TAG = os.path.splitext(os.path.basename(CKPT))[0].replace(".", "p")
REF = os.environ.get("RWKVQ_KL_REF", f"/tmp/kl_ref_{TAG}_{NSEQ}x{SEQLEN}.npy")
OUT = os.environ.get("RWKVQ_KL_OUT", f"/tmp/kl_subgroups_{TAG}.json")
BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 512)]

SUBSETS = {
    "proj_r":     r"att\.receptance\.weight$",
    "proj_k":     r"att\.key\.weight$",
    "proj_v":     r"att\.value\.weight$",
    "proj_o":     r"att\.output\.weight$",
    "cmix_key":   r"ffn\.key\.weight$",
    "cmix_value": r"ffn\.value\.weight$",
    "emb":        r"^emb\.weight$",
    "head":       r"^head\.weight$",
    "layer0":     r"^blocks\.0\.",
    "vfirst":     r"^blocks\.0\.att\.value\.weight$",
    "lora":       r"att\.[wavg][12]$",
}


def mem(tag):
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True).stdout.strip()
    used = sw.split("used =")[1].split()[0] if "used =" in sw else "?"
    mp = subprocess.run(["memory_pressure", "-Q"],
                        capture_output=True, text=True).stdout
    free = next((l.split(":")[1].strip() for l in mp.splitlines()
                 if "free percentage" in l), "?")
    print(f"  [mem/{tag}] своп {used}, свободно {free}", flush=True)


def load_data():
    blob = torch.load(CORPUS)
    tok = blob["tokens"] if isinstance(blob, dict) else blob
    langs = list(blob.get("lang", []))[:NSEQ] if isinstance(blob, dict) else []
    return tok[:NSEQ, :SEQLEN].contiguous().to("mps"), langs


def prequantize(model, cfg, pred, verbose=True):
    """Как comp.prequantize, но с ФИЛЬТРОМ по ключу.

    Фильтр -- единственное отличие; сам список точек берётся у
    ablate_sym_composite, чтобы он не разъехался с forward (закон 15).
    """
    done, t0 = {}, time.time()
    for obj, attr, group, key in comp.quant_points(model):
        if not pred(key):
            continue
        w = getattr(obj, attr)
        if w is None:
            continue
        out = fake_quant.q(w, group, cfg, key)
        if out is not w:
            setattr(obj, attr, out.to(w.dtype))
            done[group] = done.get(group, 0) + 1
        del w, out
    if verbose:
        print(f"  квантовано {sum(done.values())} тензоров за "
              f"{time.time()-t0:.0f}s, по группам {done}", flush=True)
    return done


@torch.no_grad()
def logits_of(model, data):
    """Логиты по ОДНОЙ последовательности за раз: [T-1, V] float32 на CPU.

    По одной, а не батчем: [B, 511, 65536] в fp32 -- это 134 МБ на
    последовательность, и батч на 2.9B кладёт машину."""
    for i in range(data.shape[0]):
        lg = model.forward(data[i:i + 1, :-1])
        out = lg[0].float().cpu().numpy()
        del lg
        # Закон 21: на 2.9B MPS может исчерпать память и уронить command
        # buffer, НЕ подняв исключения -- тензоры возвращаются нулевыми.
        # По ppl это выглядело как ровно 1.0000; по KL выглядело бы как
        # правдоподобное большое число, то есть ещё хуже. Проверяем.
        if not np.isfinite(out).all() or float(np.abs(out).max()) == 0.0:
            raise RuntimeError(
                f"логиты последовательности {i} вырождены (нули или "
                f"не-конечные): почти наверняка MPS уронил command buffer. "
                f"Смотрите stderr на 'Insufficient Memory'.")
        yield out


@torch.no_grad()
def trace_of(model, data1):
    """residual stream после каждого блока для ОДНОЙ последовательности."""
    tr = []
    model.forward(data1[:, :-1], trace=tr)
    return [t[0] for t in tr]


def build_ref():
    if os.path.exists(REF):
        print(f"эталон уже есть: {REF}")
        return
    data, _ = load_data()
    T, V = data.shape[1] - 1, 65536
    print(f"эталон bf16 -> {REF}  [{NSEQ}, {T}, {V}] float32 = "
          f"{NSEQ*T*V*4/1e9:.2f} ГБ", flush=True)
    mem("старт")
    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    mm = np.lib.format.open_memmap(REF, mode="w+", dtype=np.float32,
                                   shape=(NSEQ, T, V))
    for i, lg in enumerate(logits_of(model, data)):
        mm[i] = lg
        print(f"  {i+1}/{NSEQ}", flush=True)
    mm.flush()
    del mm, model
    mem("финиш")


def kl_stats(ref_row, got):
    """KL(P_bf16 ‖ Q) на каждый токен, в натах.

    Считается в float64 по стабильной формуле: KL = Σ p (lp - lq), где
    lp/lq -- log_softmax. Материализовать softmax целиком нельзя (у
    [511, 65536] это 134 МБ на копию), поэтому идём по позициям чанками.
    """
    out = np.empty(ref_row.shape[0], dtype=np.float64)
    top = np.empty(ref_row.shape[0], dtype=bool)
    for a in range(0, ref_row.shape[0], 64):
        b = min(a + 64, ref_row.shape[0])
        P, Q = ref_row[a:b].astype(np.float64), got[a:b].astype(np.float64)
        lp = P - np.log(np.exp(P - P.max(1, keepdims=True)).sum(1, keepdims=True)) - P.max(1, keepdims=True)
        lq = Q - np.log(np.exp(Q - Q.max(1, keepdims=True)).sum(1, keepdims=True)) - Q.max(1, keepdims=True)
        p = np.exp(lp)
        out[a:b] = (p * (lp - lq)).sum(1)
        top[a:b] = P.argmax(1) == Q.argmax(1)
    return out, top


def boot_ci(per_seq, iters=20000, seed=0):
    """95% CI бутстрэпом ПО ПОСЛЕДОВАТЕЛЬНОСТЯМ -- разброс корпуса
    (одни чанки просто труднее) сокращается, меряется разброс эффекта."""
    rng = np.random.default_rng(seed)
    a = np.asarray(per_seq, dtype=np.float64)
    idx = rng.integers(0, len(a), size=(iters, len(a)))
    m = a[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run(name, only, exc, trace):
    cfg = comp.CONFIGS[name]()
    if name != "bf16" and not os.path.exists(comp.ACT):
        raise SystemExit(f"нет статистики {comp.ACT} (закон 15)")
    if not os.path.exists(REF):
        raise SystemExit(f"нет эталона {REF}: сначала --ref")

    tag = name + (f" --only {only}" if only else "") + \
        (f" --except {exc}" if exc else "")
    print(f"=== {os.path.basename(CKPT)} / {tag} ===", flush=True)
    mem("старт")
    data, langs = load_data()
    ref = np.load(REF, mmap_mode="r")

    if only:
        rx = re.compile(SUBSETS[only])
        pred = lambda k: rx.search(k) is not None            # noqa: E731
    elif exc:
        rx = re.compile(SUBSETS[exc])
        pred = lambda k: rx.search(k) is None                # noqa: E731
    else:
        pred = lambda k: True                                # noqa: E731

    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    prequantize(model, cfg, pred)
    mem("после кванта")

    t0 = time.time()
    per_seq, per_bucket, top_ok, n_tok = [], {b: [] for b in BUCKETS}, 0, 0
    for i, lg in enumerate(logits_of(model, data)):
        kl, top = kl_stats(np.asarray(ref[i]), lg)
        per_seq.append(float(kl.mean()))
        for b in BUCKETS:
            sel = kl[b[0]:b[1]]
            if sel.size:
                per_bucket[b].append(float(sel.mean()))
        top_ok += int(top.sum()); n_tok += top.size
        del lg, kl, top
    lo, hi = boot_ci(per_seq)
    res = {"kl": float(np.mean(per_seq)), "ci": [lo, hi],
           "per_seq": per_seq,
           "by_pos": {f"{b[0]}-{b[1]}": float(np.mean(v))
                      for b, v in per_bucket.items() if v},
           "top1_agree": top_ok / n_tok, "seconds": round(time.time() - t0)}
    print(f"KL = {res['kl']:.6f} нат/токен  95% CI [{lo:.6f}; {hi:.6f}]")
    print("по позиции: " + "  ".join(f"{k} {v:.6f}"
                                     for k, v in res["by_pos"].items()))
    print(f"top-1 совпадает: {100*res['top1_agree']:.3f}%")

    if trace:
        # Трассы снимаются ПОСЛЕДОВАТЕЛЬНО, а не двумя моделями сразу:
        # две копии 1.5B на 16 ГБ ещё живут, 2.9B уже нет, а второй
        # прогон одной последовательности стоит секунды.
        tr_q = trace_of(model, data[:1])
        del model
        base = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
        tr_b = trace_of(base, data[:1])
        del base
        rel = [float((b - qq).norm() / b.norm()) for b, qq in zip(tr_b, tr_q)]
        res["layer_rel"] = rel
        print("‖Δx‖/‖x‖ по слоям: "
              + "  ".join(f"{l}:{v:.1e}" for l, v in enumerate(rel)))

    doc = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    doc.update(ckpt=CKPT, ref=REF, nseq=NSEQ, seqlen=SEQLEN)
    doc["rows"][tag] = res
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    mem("финиш")
    return 0


def report():
    if not os.path.exists(OUT):
        print("нет данных")
        return
    doc = json.load(open(OUT))
    rows = doc["rows"]
    # «полный» -- это конфиг БЕЗ фильтра и не bf16. Первая версия брала
    # первую строку без "--", то есть bf16 с KL=0, и вклад для --except
    # печатался как минус собственный KL: число выглядело осмысленным и
    # было бессмысленным.
    full = next((v for k, v in rows.items()
                 if "--" not in k and k != "bf16"), None)
    print(f"\n{os.path.basename(doc['ckpt'])}, {doc['nseq']}x{doc['seqlen']}, "
          f"KL(bf16 ‖ квантованная), нат/токен")
    print(f"\n{'конфиг':38s} {'KL':>10s} {'95% CI':>22s} {'вклад':>10s} "
          f"{'% полного':>10s} {'top-1':>8s}")
    for k, v in rows.items():
        ci = f"[{v['ci'][0]:.5f}; {v['ci'][1]:.5f}]"
        # для --except вклад подмножества = KL(всё) - KL(всё кроме него)
        d, sh = "", ""
        if "--except" in k and full:
            c = full["kl"] - v["kl"]
            d, sh = f"{c:+.6f}", f"{100*c/full['kl']:+9.1f}%"
        elif "--only" in k:
            d = f"{v['kl']:.6f}"
            if full:
                sh = f"{100*v['kl']/full['kl']:9.1f}%"
        print(f"{k:38s} {v['kl']:10.6f} {ci:>22s} {d:>10s} {sh:>10s} "
              f"{100*v['top1_agree']:7.3f}%")
    if full:
        tot = sum(v["kl"] for k, v in rows.items() if "--only" in k)
        print(f"\nсумма изолированных {tot:.6f} против композита "
              f"{full['kl']:.6f} -- {100*full['kl']/tot:.0f}%: ошибки групп "
              f"частично ГАСЯТ друг друга, складывать их нельзя")
    print("\nвклад: для --except это KL(всё) − KL(всё кроме подмножества), "
          "то есть сколько уносит именно оно;\nдля --only -- его собственный "
          "KL при всём остальном в bf16.")


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--report":
        report()
        print(f"\nподмножества: {' '.join(SUBSETS)}")
        print(f"конфиги: {' '.join(comp.CONFIGS)}")
        return 0
    if a[0] == "--ref":
        build_ref()
        return 0
    name, only, exc, trace = a[0], None, None, False
    i = 1
    while i < len(a):
        if a[i] == "--only":
            only = a[i + 1]; i += 2
        elif a[i] == "--except":
            exc = a[i + 1]; i += 2
        elif a[i] == "--trace":
            trace = True; i += 1
        else:
            raise SystemExit(f"не понял аргумент {a[i]}")
    for s in (only, exc):
        if s and s not in SUBSETS:
            raise SystemExit(f"нет подмножества {s}; есть: {list(SUBSETS)}")
    return run(name, only, exc, trace)


if __name__ == "__main__":
    sys.exit(main())
