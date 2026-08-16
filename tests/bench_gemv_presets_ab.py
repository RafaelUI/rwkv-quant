"""ПОЛОСА В GEMV-ФАЗЕ У ДВУХ ПРЕСЕТОВ, ОБЕ МОДЕЛИ В ОДНОМ ПРОЦЕССЕ.

Разделяет две версии, записанные 16.08 после внешнего счётчика полосы
(REDUCTION 80 ГБ/с, COMPRESSION не выше 73):

  (1) виновата ПОСТОЯННАЯ ЧАСТЬ -- она одинакова в миллисекундах, а шаг у
      COMPRESSION короче, поэтому средняя по шагу полоса падает сама.
      Лечится оп-каунтом;
  (2) виновата РАСПАКОВКА -- на 4-5 битах она дороже в пересчёте на байт.
      Лечится кернелем.

`bench_step_decompose` отвечает на это вычитанием, но по ПРОЦЕССУ на
пресет, а межпроцессный разброс тут втрое-вчетверо больше
внутрипроцессного (закон 24) и на безвентиляторной машине второй прогон
идёт на прогретой (закон 25): в двух порядках REDUCTION дал 94.0 и 88.2
ГБ/с. Здесь обе модели живут одновременно и GEMV-фазы ЧЕРЕДУЮТСЯ, то есть
защищено ОТНОШЕНИЕ -- ровно то, что и спрашивается.

Меряется только GEMV-фаза (та же цепочка, что в bench_step_decompose):
r/k/v/o + cmix key/value на каждом слое плюс голова. Постоянная часть сюда
не входит вовсе, поэтому версия (1) на этот замер влиять не может -- если
полоса всё равно разойдётся, остаётся (2).

    python bench_gemv_presets_ab.py [A.rwkvq] [B.rwkvq] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402
from trace_decode_steady import decode_traffic_mb  # noqa: E402

A = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
B = sys.argv[2] if len(sys.argv) > 2 else "/tmp/champion_v2.rwkvq"
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 9
REPS, WARM = 7, 3


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    return float(num) * (1024 if unit == "G" else 1)


def gemv_fn(model, x):
    def run():
        y = x
        for b in model.blocks:
            t, c = b.tmix, b.cmix
            y = (t.r_proj(y) + t.k_proj(y) + t.v_proj(y) + t.o_proj(y)) * 0.25
            y = c.value(c.key(y))
        return model.head(y)
    return run


def main():
    models, fns, traf = {}, {}, {}
    for name, path in (("A " + os.path.basename(A), A),
                       ("B " + os.path.basename(B), B)):
        m = QuantRWKV7(load_raw(path))
        D = int(m.emb_weight.shape[1])
        x = mx.array(np.random.randn(1, D).astype(np.float32))
        mx.eval(x)
        models[name] = m
        fns[name] = gemv_fn(m, x)
        traf[name] = decode_traffic_mb(m)
    names = list(fns)

    for n in names:
        for _ in range(WARM):
            mx.eval(fns[n]())
    mx.synchronize()

    sw0 = swap_mb()
    acc = {n: [] for n in names}
    for _ in range(ROUNDS):
        for n in names:                      # чередование внутри раунда
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                mx.eval(fns[n]())
                mx.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            acc[n].append(float(np.median(ts)))
    sw1 = swap_mb()

    print(f"своп {sw0:.0f} -> {sw1:.0f} МБ"
          + ("   *** НЕДЕЙСТВИТЕЛЕН: рос во время замера (закон 11) ***"
             if sw1 > sw0 + 0.5 else "   (не рос)"))
    print(f"\n{'пресет':<28}{'GEMV мс':>10}{'разброс':>9}{'трафик МБ':>11}"
          f"{'ГБ/с':>9}")
    bw = {}
    for n in names:
        med = float(np.median(acc[n]))
        sp = 100 * (max(acc[n]) - min(acc[n])) / med
        bw[n] = traf[n] / med
        print(f"{n:<28}{med:>10.3f}{sp:>8.1f}%{traf[n]:>11.1f}{bw[n]:>9.1f}")

    a, b = names
    print(f"\nотношение полос A/B: {bw[a]/bw[b]:.3f}x")
    print("ВЕРДИКТ: " + (
        "полоса в GEMV-фазе РАЗЛИЧАЕТСЯ -> виновата распаковка, лечится "
        "кернелем" if abs(bw[a] / bw[b] - 1) > 0.04 else
        "полоса в GEMV-фазе ОДИНАКОВА -> виновата постоянная часть, лечится "
        "оп-каунтом"))
    print("Порог 4% взят как удвоенный внутрипроцессный разброс; если "
          "фактический разброс выше -- вердикт не читать.")


if __name__ == "__main__":
    main()
