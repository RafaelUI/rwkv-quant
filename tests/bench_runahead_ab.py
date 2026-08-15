"""СКОЛЬКО ШАГА СТОИТ СИНХРОНИЗАЦИЯ С CPU, А НЕ РАБОТА GPU.

ВОПРОС, НА КОТОРЫЙ ЭТО ОТВЕЧАЕТ. В Metal System Trace видно
`CPU to GPU Latency` 2.8 мс в среднем и до 4.3 мс на отдельных буферах.
Соблазн прочитать это как «столько мы теряем» велик, но счётчик меряет
время от commit до старта буфера на GPU, а оно складывается из ДВУХ
разных вещей:

  (а) НАКЛАДНЫЕ ПОСТАНОВКИ -- CPU кодирует и коммитит, GPU ждёт. Это
      потеря, и она снимается тем, что CPU забегает вперёд.
  (б) ОЧЕРЕДЬ -- буфер ждёт, потому что GPU занят предыдущими. Это НЕ
      потеря, это признак того, что GPU насыщен, и «уменьшить до 1 мс»
      тут означает лишь «поставить буфер позже».

По одному счётчику (а) от (б) не отличить, и в этом всё дело. Отличает
их прямой замер: если CPU дать ЗАБЕЖАТЬ ВПЕРЁД, то (а) исчезает, а (б)
остаётся. Декод -- цепочка зависимостей по состоянию, поэтому GPU в
любом случае считает шаги последовательно; меняется только то, успел ли
CPU поставить работу заранее.

КАК ЗАБЕГ ДЕЛАЕТСЯ. MLX ленив: если не звать `mx.eval`, шаги
накапливаются в граф, и CPU уходит вперёд на K шагов. Токен при этом
остаётся ленивым массивом и подаётся в следующий шаг как есть, то есть
последовательность работы та же с точностью до бита -- это проверяется
сравнением выданных токенов, а не предполагается.

ЧТО ОЗНАЧАЕТ РЕЗУЛЬТАТ:
  - время падает с ростом K  -> на критическом пути стоит постановка,
    и выигрыш ограничен сверху этой разницей. Практически применимо:
    жадный декод не обязан синхронизироваться на КАЖДОМ токене, ему
    достаточно раз в K ради стоп-условия.
  - время не падает         -> GPU насыщен, 2.8 мс есть очередь, и
    оптимизировать надо объём работы, а не задержку.

Закон 24 (свип -- тоже замер скорости): значения K ЧЕРЕДУЮТСЯ внутри
раунда, а не гоняются подряд.

    python tests/bench_runahead_ab.py [model.rwkvq] [раундов]
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
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
KS = (1, 2, 4, 8, 16)
NSTEP = 32


def swap_mb():
    env = dict(os.environ, LC_ALL="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True, env=env).stdout.replace("=", " ").split()
    return float(out[out.index("used") + 1].rstrip("M").replace(",", "."))


def main():
    qm.FUSE = True
    model = qm.QuantRWKV7(load_raw(PATH))
    fn = mx.compile(model.forward_stateful)
    prompt = mx.array(np.array([[1, 2, 3, 4]], dtype=np.int32))

    def burst(K, n=NSTEP, collect=False):
        """n шагов декода с синхронизацией раз в K. Возвращает мс/токен
        и, если попросили, выданные токены -- чтобы убедиться, что при
        разных K считалось ОДНО И ТО ЖЕ."""
        st = model.init_state(1)
        logits, st = fn(prompt, st, True)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok, *st)
        mx.synchronize()
        toks = []
        t0 = time.perf_counter()
        for i in range(n):
            logits, st = fn(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            if collect:
                toks.append(tok)
            if (i + 1) % K == 0:
                mx.eval(tok, *st)      # <- единственное отличие вариантов
        mx.eval(tok, *st)
        mx.synchronize()
        dt = (time.perf_counter() - t0) / n * 1e3
        return dt, ([int(np.array(t).reshape(-1)[0]) for t in toks]
                    if collect else None)

    # --- контроль: при всех K выдаётся одна и та же последовательность
    ref = None
    for K in KS:
        _, toks = burst(K, n=16, collect=True)
        if ref is None:
            ref = toks
        elif toks != ref:
            bad = sum(a != b for a, b in zip(toks, ref))
            print(f"ПРОВАЛ: при K={K} токены разошлись в {bad} из 16 позиций "
                  f"-- варианты считают РАЗНОЕ, замер недействителен")
            sys.exit(1)
    print(f"контроль: все K дают одинаковые 16 токенов ({ref[:6]}...)")

    for _ in range(2):                      # прогрев всех веток
        for K in KS:
            burst(K, n=8)

    sw0 = swap_mb()
    res = {K: [] for K in KS}
    for _ in range(ROUNDS):
        for K in KS:                        # ЧЕРЕДОВАНИЕ (закон 24)
            res[K].append(burst(K)[0])
    sw1 = swap_mb()

    base = float(np.median(res[1]))
    print(f"\n{'K (шагов на синхр.)':>22} | {'мс/ток':>8} | {'ток/с':>7} | "
          f"{'выигрыш':>9} | разброс")
    for K in KS:
        m = float(np.median(res[K]))
        spread = 100 * (max(res[K]) - min(res[K])) / m
        print(f"{K:>22} | {m:8.2f} | {1000/m:7.1f} | "
              f"{base - m:+8.2f} мс | {spread:.1f}%")
    print(f"\nсвоп: {sw0:.0f} -> {sw1:.0f} МБ (дельта {sw1-sw0:+.0f})")
    print("K=1 -- это то, как декодирует настоящий сэмплер: синхронизация "
          "на каждом токене.")


if __name__ == "__main__":
    main()
