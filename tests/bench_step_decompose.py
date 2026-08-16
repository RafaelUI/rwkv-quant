"""Из чего состоит шаг декода: GEMV против всего остального.

ЗАЧЕМ. Metal-трасса 13.08 показала Instruction Throughput Limiter 92% при
целочисленной, а не плавающей нагрузке -- то есть шаг упирается в ВЫДАЧУ
инструкций. Но средние по шагу смешивают ~1000 запусков разной природы, и
по ним нельзя отделить «большие GEMV плохи» от «мелочь плоха». Per-kernel
атрибуция в Metal System Trace недоступна (там видны команды буфера, а не
имена ядер; для имён нужен отдельный GPU frame capture), поэтому
разложение делается здесь -- вычитанием, а не профилировщиком.

МЕТОД. В одном процессе чередуются:
  full   -- настоящий шаг модели;
  gemv   -- ТОЛЬКО линейные слои (r/k/v/o + cmix key/value на каждом слое
            плюс голова), в том же порядке и на тех же буферах;
  gemv-1 -- то же без головы, чтобы отделить её вклад;
  empty  -- пустая операция, пол диспетчеризации.
Разность full - gemv и есть бюджет «всего остального»: LoRA-ветки, WKV,
нормы, лерпы, bonus, gate и лишние запуски.

Чередование обязательно (закон 1), своп фиксируется (закон 11), окно
короткое -- троттлинг съедает абсолюты за минуту (закон 25).

    python tests/bench_step_decompose.py [конфиг]
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402

import ablate_sym_composite as comp  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "reduction_sym_head8"
ROUNDS, REPS, WARM = 9, 7, 3


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    return float(num) * (1024 if unit == "G" else 1)


def bench(fn):
    for _ in range(WARM):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    sw0 = swap_mb()
    if NAME.endswith(".rwkvq"):
        # Готовый файл вместо сборки из .pth. Так меряется РОВНО тот
        # артефакт, который лежит на диске (а не дельта-конфиг поверх
        # живого пресета -- закон 30), и не нужен воркспейс квантования на
        # 7.7 ГБ. Для СКОРОСТИ это эквивалентно: раскладка и размеры те же.
        from rwkv_quant.formats.reader import load_raw
        model = QuantRWKV7(load_raw(NAME))
        gc.collect()
        mx.clear_cache()
    else:
        cfg = comp.CONFIGS[NAME]()
        sd = torch.load(comp.CKPT, map_location="cpu", mmap=True)
        n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        emb = sd["emb.weight"]
        r_k = next(v for k, v in sd.items() if k.endswith("r_k"))
        meta = dict(naming="world", n_layer=n_layer, n_embd=int(emb.shape[1]),
                    vocab_size=int(emb.shape[0]), head_size=int(r_k.shape[-1]))
        tensors = {k: quantize_tensor(k, w, cfg, real_gw=True) for k, w in sd.items()}
        model = QuantRWKV7(QuantizedCheckpoint(
            tensors=tensors, config_repr=repr(cfg), **meta))
        del tensors, sd
        gc.collect()
        mx.clear_cache()

    D = int(model.emb_weight.shape[1])
    x = mx.array(np.random.randn(1, D).astype(np.float32))
    st = model.init_state(1)
    tok = mx.array(np.array([1], dtype=np.int32))
    mx.eval(x)

    # Своп на ГРАНИЦЕ замера, а не от старта процесса: сборка модели из
    # .pth держит воркспейс грид-поиска и законно выталкивает чужие
    # страницы. Замер портит пейджинг ВО ВРЕМЯ измерения (см. уточнение к
    # закону 11 в bench_sym_e2e_ab).
    sw_bench = swap_mb()
    state = {"st": st, "tok": tok}

    def full():
        logits, state["st"] = model.step(state["tok"][None], state["st"])
        state["tok"] = mx.argmax(logits[:, -1], axis=-1)
        return state["tok"]

    def gemv(with_head=True):
        def run():
            y = x
            for b in model.blocks:
                t, c = b.tmix, b.cmix
                y = (t.r_proj(y) + t.k_proj(y) + t.v_proj(y) + t.o_proj(y)) * 0.25
                y = c.value(c.key(y))
            return model.head(y) if with_head else y
        return run

    cases = {"full": full, "gemv+head": gemv(True), "gemv": gemv(False),
             "empty": lambda: x + 1.0}
    acc = {k: [] for k in cases}
    for _ in range(ROUNDS):
        for k, fn in cases.items():
            acc[k].append(bench(fn))

    sw1 = swap_mb()
    src = NAME if NAME.endswith(".rwkvq") else os.path.basename(comp.CKPT)
    print(f"=== {NAME} на {src} ===")
    print(f"своп: старт {sw0:.1f} -> вход в замер {sw_bench:.1f} -> "
          f"конец {sw1:.1f} МБ"
          + ("   *** НЕДЕЙСТВИТЕЛЕН: рос ВО ВРЕМЯ замера (закон 11) ***"
             if sw1 > sw_bench + 0.5 else "   (во время замера не рос)"))
    med = {k: float(np.median(v)) for k, v in acc.items()}
    print(f"\n{'что':12s} {'мс':>8s} {'разброс':>8s}")
    for k in cases:
        sp = 100 * (max(acc[k]) - min(acc[k])) / med[k]
        print(f"{k:12s} {med[k]:8.3f} {sp:7.1f}%")
    rest = med["full"] - med["gemv+head"]
    print(f"\nGEMV (24 слоя x 6 + голова): {med['gemv+head']:.3f} мс "
          f"= {100*med['gemv+head']/med['full']:.0f}% шага")
    print(f"ВСЁ ОСТАЛЬНОЕ (LoRA, WKV, нормы, лерпы, bonus, gate, лишние "
          f"запуски): {rest:.3f} мс = {100*rest/med['full']:.0f}%")
    print(f"голова отдельно: {med['gemv+head']-med['gemv']:.3f} мс")
    print(f"пол диспетчеризации (одна операция): {med['empty']:.3f} мс")

    # ГБ/с В ФАЗЕ САМИХ ЧТЕНИЙ -- то, ради чего этот замер и ставится на
    # двух пресетах. Внешний счётчик 16.08 дал 80 ГБ/с на REDUCTION и не
    # выше 73 на COMPRESSION, и две версии («распаковка на 4-5 битах
    # дороже» против «постоянная часть та же в мс, а шаг короче») тут
    # разделяются: если полоса в GEMV-фазе у пресетов ОДИНАКОВА -- виновата
    # постоянная часть и лечится оп-каунтом; если у COMPRESSION ниже --
    # виновата распаковка и лечится кернелем.
    try:
        from trace_decode_steady import decode_traffic_mb
        traf = decode_traffic_mb(model)
        print(f"\nтрафик за токен (обход живых буферов): {traf:.1f} МБ")
        print(f"полоса в GEMV-фазе:  {traf / med['gemv+head']:.1f} ГБ/с")
        print(f"полоса по шагу целиком: {traf / med['full']:.1f} ГБ/с "
              f"(так её видит внешний счётчик)")
        print(f"постоянная часть (не-GEMV): {rest:.3f} мс = "
              f"{100*rest/med['full']:.0f}% шага")
    except Exception as e:  # noqa: BLE001
        print(f"\nтрафик не посчитан: {type(e).__name__}: {e}")
    print(f"\nЕсли «остальное» -- это фьюзабельная мелочь, то его сокращение "
          f"вдвое даёт {med['full']-rest/2:.2f} мс/ток = "
          f"{1000/(med['full']-rest/2):.1f} ток/с против "
          f"{1000/med['full']:.1f} сейчас.")


if __name__ == "__main__":
    main()
