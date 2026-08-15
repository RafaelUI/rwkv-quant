"""ГРАНУЛЯРНОСТЬ КОМАНДНЫХ БУФЕРОВ MLX: сколько шага стоит их дробление.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ЗАБЕГА ВПЕРЁД (`bench_runahead_ab.py`). Тот замер
проверял, ждёт ли GPU, пока CPU поставит работу, -- и ответил, что нет.
Но между ДВУМЯ командными буферами есть зазор независимо от того, кто
кого ждал: буфер завершается, GPU сообщает об этом, начинается
следующий. MLX режет граф на буферы по числу операций
(`MLX_MAX_OPS_PER_BUFFER`) и по объёму (`MLX_MAX_MB_PER_BUFFER`), а в
шаге декода порядка тысячи примитивов -- то есть буферов может быть
СОТНЯ на токен, и тогда зазоры складываются в миллисекунды.

`MLX_METAL_FAST_SYNCH` -- третья ручка: меняет способ ожидания
завершения буфера.

ПОЧЕМУ ЭТО ЗАПУСК ПОДПРОЦЕССОВ. Переменные читаются при инициализации
Metal-бэкенда, в одном процессе их не переключить. Значит закон 1
(чередование) в чистом виде недоступен, и остаётся его ослабленная
форма: значения гоняются ПО ОЧЕРЕДИ раунд за раундом, а не подряд, и
рядом печатается разброс по раундам. Закон 24 предупреждает ровно про
это: последовательный свип рисует эффекты, которых нет, -- поэтому
вывод делать только если отрыв БОЛЬШЕ разброса.

    python tests/bench_cmdbuf_ab.py [model.rwkvq] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
        else "/tmp/reduction_sym_head8.rwkvq")

# (метка, переменные окружения)
VARIANTS = [
    ("умолчание", {}),
    ("ops=100", {"MLX_MAX_OPS_PER_BUFFER": "100"}),
    ("ops=1000", {"MLX_MAX_OPS_PER_BUFFER": "1000"}),
    ("fast_synch", {"MLX_METAL_FAST_SYNCH": "1"}),
    ("ops=1000+fs", {"MLX_MAX_OPS_PER_BUFFER": "1000",
                     "MLX_METAL_FAST_SYNCH": "1"}),
]


def child(path):
    import mlx.core as mx
    import numpy as np
    import rwkv_quant.backends.metal.quant_model as qm
    from rwkv_quant.formats.reader import load_raw

    qm.FUSE = True
    model = qm.QuantRWKV7(load_raw(path))
    fn = mx.compile(model.forward_stateful)
    prompt = mx.array(np.array([[1, 2, 3, 4]], dtype=np.int32))

    def burst(n):
        st = model.init_state(1)
        logits, st = fn(prompt, st, True)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok, *st)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, st = fn(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(tok)
        mx.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    burst(12)                                  # прогрев
    print(f"MS {min(burst(24) for _ in range(3)):.4f}", flush=True)


def main():
    import numpy as np
    rounds = 4
    for a in sys.argv[1:]:
        if a.isdigit():
            rounds = int(a)
    res = {lab: [] for lab, _ in VARIANTS}
    me = os.path.abspath(__file__)
    for r in range(rounds):
        for lab, ev in VARIANTS:            # ПО ОЧЕРЕДИ, не подряд
            env = dict(os.environ, LC_ALL="C", **ev)
            out = subprocess.run(
                [sys.executable, me, PATH, "--child"],
                env=env, capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("MS ")]
            if not line:
                print(f"ПРОВАЛ {lab}: {out.stderr.strip()[-300:]}")
                sys.exit(1)
            res[lab].append(float(line[0][3:]))
        print(f"раунд {r+1}/{rounds} готов", flush=True)

    base = float(np.median(res["умолчание"]))
    print(f"\n{'вариант':>14} | {'мс/ток':>8} | {'ток/с':>7} | "
          f"{'против умолч.':>14} | разброс")
    for lab, _ in VARIANTS:
        m = float(np.median(res[lab]))
        spread = 100 * (max(res[lab]) - min(res[lab])) / m
        print(f"{lab:>14} | {m:8.2f} | {1000/m:7.1f} | "
              f"{base - m:+9.2f} мс | {spread:.1f}%")
    print("\nВывод делать только если отрыв больше разброса (закон 24).")


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(sys.argv[1])
    else:
        main()
