"""ПРЕФИЛЛ КУСКАМИ ФИКСИРОВАННОЙ ФОРМЫ: как получить компилированную
скорость на ПРОИЗВОЛЬНОЙ длине промпта.

ЗАДАЧА. `mx.compile` даёт префиллу +35% (344 -> 533 ток/с, замерено
`bench_compile_ab`), но кеш ключуется ФОРМОЙ: первый вызов на новой длине
стоит ~110 мс трассировки. В чате длины произвольные, значит платится это
почти на каждом промпте.

Выход напрашивается: гонять префилл кусками ОДНОЙ длины с переносом
состояния. Тогда трассировка ровно одна на всё время жизни процесса,
какой бы промпт ни пришёл. Вопрос только в цене: короче кусок -- меньше
работы на запуск и хуже амортизация.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ЗАМЕРА 13.08. Тогда разбиение промпта на куски
проверялось как гипотеза про ПАМЯТЬ (не помогло: −3% пика ценой падения
скорости вдвое) -- и делалось на СЫРОМ пути с `mx.eval` после каждого
куска, то есть с намеренным обрывом графа. Здесь всё наоборот: путь
компилированный, а граф НЕ рвётся (eval только в конце), поэтому куски
конвейеризуются. Это другой замер, а не повторение.

Контроль обязателен: куски обязаны давать те же логиты, что один кусок,
иначе сравниваются разные вычисления.

    python tests/bench_prefill_chunk_ab.py [model.rwkvq] [T] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
CHUNKS = (64, 128, 256)


def swap_mb():
    env = dict(os.environ, LC_ALL="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True, env=env).stdout.replace("=", " ").split()
    return float(out[out.index("used") + 1].rstrip("M").replace(",", "."))


def main():
    model = qm.QuantRWKV7(load_raw(PATH))
    comp = mx.compile(model.forward_stateful)
    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))

    def whole():
        st = model.init_state(1)
        logits, st = comp(prompt, st, True)
        mx.eval(logits)
        return logits

    def chunked(C, break_graph=False):
        def run():
            st = model.init_state(1)
            logits = None
            for a in range(0, T, C):
                logits, st = comp(prompt[:, a:a + C], st, True)
                if break_graph:
                    mx.eval(logits, *[s for s in st])
            mx.eval(logits)
            return logits
        return run

    # --- контроль: куски считают то же, что один кусок
    ref = np.array(whole().astype(mx.float32))
    for C in CHUNKS:
        got = np.array(chunked(C)().astype(mx.float32))
        rel = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
        if rel > 3e-3:
            print(f"ПРОВАЛ: куски по {C} дают relmax {rel:.2e} против одного "
                  f"куска -- сравниваются разные вычисления")
            sys.exit(1)
        print(f"контроль C={C}: relmax к одному куску {rel:.2e}")

    variants = [("один кусок", whole)]
    for C in CHUNKS:
        variants.append((f"куски {C}", chunked(C)))
    variants.append((f"куски {CHUNKS[-1]} + eval", chunked(CHUNKS[-1], True)))

    for _ in range(2):                          # прогрев всех трассировок
        for _, f in variants:
            f()
    mx.synchronize()

    sw0 = swap_mb()
    res = {lab: [] for lab, _ in variants}
    for _ in range(ROUNDS):
        for lab, f in variants:                 # ЧЕРЕДОВАНИЕ (закон 1)
            mx.synchronize()
            t0 = time.perf_counter()
            f()
            mx.synchronize()
            res[lab].append((time.perf_counter() - t0) * 1e3)
    sw1 = swap_mb()

    base = float(np.median(res["один кусок"]))
    print(f"\n{'вариант':>22} | {'мс':>8} | {'ток/с':>7} | {'против':>8} | разброс")
    for lab, _ in variants:
        m = float(np.median(res[lab]))
        spread = 100 * (max(res[lab]) - min(res[lab])) / m
        print(f"{lab:>22} | {m:8.1f} | {T/m*1e3:7.1f} | "
              f"{100*(base-m)/base:+7.1f}% | {spread:.1f}%")
    print(f"\nсвоп: {sw0:.0f} -> {sw1:.0f} МБ (дельта {sw1-sw0:+.0f})")
    print("«один кусок» требует трассировки на КАЖДУЮ длину промпта (~110 мс); "
          "куски -- одну на процесс.")


if __name__ == "__main__":
    main()
