"""Сколько модель занимает в ОПЕРАТИВНОЙ памяти и сколько трафика едет за
токен -- для пресета с sym против нынешнего.

ДВА ЧИСЛА, КОТОРЫЕ ПУТАЮТ. Размер файла и резидентная память -- разные
вещи (замер 04.08: 1.85x от файла), а скорость декода определяется НИ
ТЕМ, НИ ДРУГИМ, а ТРАФИКОМ за токен: `emb` -- это gather одной строки,
и вся таблица за токен не читается. Считать «файл / полоса» -- значит
завышать требуемую полосу на размер emb.

ПОЧЕМУ ЗДЕСЬ НЕТ mx.get_active_memory. Метрики фреймворков для unified
memory врут (закон 11), и на них тут полагаться нельзя. Резидентность
берётся из `vm_stat` (системный счётчик), пик -- из
`/usr/bin/time -l` -> `peak memory footprint` (закон 22), а состав --
прямым обходом живых mx-массивов модели, то есть подсчётом байт, а не
опросом аллокатора.

Обход идёт только по __dict__ (см. report_inference_memory): у
GwQuantLinear старые имена -- ленивые view через __getattr__, и getattr
по ним МАТЕРИАЛИЗОВАЛ БЫ буферы, которых в памяти нет.

Один конфиг на процесс (закон 13):

    /usr/bin/time -l python tests/report_sym_memory.py reduction
    /usr/bin/time -l python tests/report_sym_memory.py reduction_sym_head8
"""
import gc
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.formats.writer import TENSOR_FIELDS, quantize_tensor  # noqa: E402
from report_inference_memory import walk  # noqa: E402

import ablate_sym_composite as comp  # noqa: E402

PAGE = 16384


def vm_used_mb():
    """Занято системой: active + wired + compressed, по vm_stat.

    Не RSS и не аллокатор MLX -- unified memory им не видна (закон 11)."""
    # LC_ALL=C по той же причине, что и у swap_mb в бенчах: разбор вывода
    # системной утилиты не должен зависеть от локали пользователя
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["vm_stat"], env=env,
                         capture_output=True, text=True).stdout
    v = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, _, n = line.partition(":")
        n = n.strip().rstrip(".")
        # первая строка vm_stat -- "Mach Virtual Memory Statistics:
        # (page size of 16384 bytes)", и она тоже содержит двоеточие
        if not n.isdigit():
            continue
        v[k.strip()] = int(n)
    pages = (v.get("Pages active", 0) + v.get("Pages wired down", 0)
             + v.get("Pages occupied by compressor", 0))
    return pages * PAGE / 1e6


def bucket(path, dtype, owners):
    """Раскладку НЕЛЬЗЯ определять по имени поля: `qblk` есть и у sb6, и у
    sym, и первая версия отчёта записала 1040 МБ sym-буферов в строку
    «sb6». Смотрим на СОСЕДЕЙ по владельцу: у sb6 рядом лежит `qsqm`, у
    sym -- `qs` и `d`. Это тот же принцип, что в codec: раскладка
    опознаётся по составу буферов, а не по имени."""
    p = path.split(".")[-1]
    owner = path.rsplit(".", 1)[0]
    sib = owners.get(owner, set())
    if "_rkv_fused" in path:
        return "ДУБЛЬ фьюза r/k/v (в память, не в трафик)"
    if p in ("qblk", "qsqm", "ddm", "qs", "d"):
        return "sb6 (K3-интерлив)" if "qsqm" in sib else "sym (интерлив)"
    if "emb_weight" in path:
        return "emb (плотная таблица)"
    if "wav_" in path or "_lora" in path.lower():
        return f"LoRA-ветки {dtype}"
    if dtype == "float32":
        return "мелочь в fp32 (нормы, token-shift)"
    return f"прочее {dtype}"


def main():
    name = sys.argv[1]
    cfg = comp.CONFIGS[name]()
    sd = torch.load(comp.CKPT, map_location="cpu", mmap=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    emb = sd["emb.weight"]
    r_k = next(v for k, v in sd.items() if k.endswith("r_k"))
    meta = dict(naming="world", n_layer=n_layer, n_embd=int(emb.shape[1]),
                vocab_size=int(emb.shape[0]), head_size=int(r_k.shape[-1]))

    tensors = {k: quantize_tensor(k, w, cfg, real_gw=True) for k, w in sd.items()}
    file_mb = sum(sum(getattr(q, f).numel() * getattr(q, f).element_size()
                      for f in TENSOR_FIELDS if getattr(q, f, None) is not None)
                  for q in tensors.values()) / 1e6
    del sd
    gc.collect()

    v0 = vm_used_mb()
    model = QuantRWKV7(QuantizedCheckpoint(tensors=tensors, config_repr=repr(cfg),
                                           **meta))
    del tensors
    gc.collect()
    mx.eval(model.emb_weight)
    mx.clear_cache()
    v1 = vm_used_mb()

    seen, arrays = set(), []
    walk(model, "", seen, arrays)
    owners = {}
    for p, _n, _dt in arrays:
        owners.setdefault(p.rsplit(".", 1)[0], set()).add(p.split(".")[-1])
    by, top = {}, {}
    for p, n, dt in arrays:
        e = by.setdefault(bucket(p, dt, owners), [0, 0])
        e[0] += n
        e[1] += 1
        key = ("blocks[]." + ".".join(p.split(".")[2:])) if "blocks" in p else p
        top[key] = top.get(key, 0) + n
    total = sum(v for v, _ in by.values()) / 1e6

    # ТРАФИК ЗА ТОКЕН -- НЕ равен ни файлу, ни резидентной памяти:
    #   - таблица emb не читается: gather одной строки;
    #   - ДУБЛЬ фьюза r/k/v читается ЛИБО он, ЛИБО отдельные r/k/v_proj,
    #     но не оба. Считать обе копии -- завысить трафик на 226 МБ на
    #     1.5B и получить «83 ГБ/с» там, где на самом деле 70.
    # Отсюда же следует, что «файл / полоса» -- неверная оценка потолка
    # скорости в обе стороны сразу.
    emb_mb = by.get("emb (плотная таблица)", [0])[0] / 1e6
    fuse_mb = by.get("ДУБЛЬ фьюза r/k/v (в память, не в трафик)", [0])[0] / 1e6
    row_mb = meta["n_embd"] * 2 / 1e6
    traffic = total - emb_mb - fuse_mb + row_mb
    from rwkv_quant.backends.metal import quant_model as qm
    # фьюз строится ЛЕНИВО, по первому входу в фьюзнутый путь (см.
    # QuantTMix._build_fused): при FUSE=False дубля в памяти нет вовсе, и
    # 0.0 МБ здесь -- это факт, а не отсутствие замера. Прежде он строился
    # безусловно и лежал мёртвым грузом (250 МБ на 1.5B) -- именно этим, а
    # не раскладкой, объяснялось прежнее "пресет с sym на 227 МБ легче".
    print(f"\n(FUSE={qm.FUSE}; дубль фьюза r/k/v в памяти: {fuse_mb:.1f} МБ"
          + ("" if fuse_mb else " -- фьюз не построен, он ленивый") + ")")

    print(f"=== {name} на {os.path.basename(comp.CKPT)} ===")
    print(f"файл (буферы .rwkvq):        {file_mb:8.1f} МБ")
    print(f"\nв памяти, обход живых массивов:")
    for k, (v, n) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"  {k:<32} {v/1e6:8.1f} МБ  ({n} массивов)")
    print(f"  {'ИТОГО':<32} {total:8.1f} МБ   = {total/file_mb:.2f}x от файла")
    # vm_stat-дельта тут нужна как грубый ориентир: между двумя
    # снимками система живёт своей жизнью и освобождает воркспейс
    # квантования, поэтому число скачет и даже бывает отрицательным.
    # Настоящая метрика -- peak memory footprint из /usr/bin/time -l
    # (закон 22), но она включает и транзиенты СБОРКИ, а не только модель.
    print(f"\nсистемно (vm_stat, active+wired+compressed): прирост за "
          f"постройку {v1-v0:+.1f} МБ (ориентир, см. комментарий)")
    print(f"\nТРАФИК ЗА ТОКЕН (без таблицы emb -- она gather):"
          f" {traffic:8.1f} МБ")
    for ms in (18.33, 18.90):
        print(f"  при {ms:5.2f} мс/ток -> {traffic/ms:6.1f} ГБ/с, "
              f"{1000/ms:5.1f} ток/с")
    print(f"\nпотолки по полосе для этого трафика:")
    for bw, tag in ((95.7, "достижимая, замерена"), (120.0, "паспортный пик")):
        ms = traffic / bw
        print(f"  {bw:5.1f} ГБ/с ({tag:22s}): {ms:5.2f} мс/ток = "
              f"{1000/ms:5.1f} ток/с")
    print("\nтоп по расходу:")
    for k, v in sorted(top.items(), key=lambda x: -x[1])[:8]:
        print(f"  {v/1e6:9.1f} МБ  {k}")


if __name__ == "__main__":
    main()
