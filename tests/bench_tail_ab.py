"""A/B хвоста TMix: FUSE_TAIL=False против True, чередованием в ОДНОМ
процессе, при FUSE=True в обеих ветках.

ЗАЧЕМ ОТДЕЛЬНЫЙ ЗАМЕР. `test_fuse_parity` меряет фьюз ЦЕЛИКОМ против
нефьюзнутого пути, поэтому вклад именно хвоста в нём не виден: между
прогонами абсолюты уплывают на 4-5% (закон 25), и разница «было +1.01,
стало +1.14» может оказаться дрейфом, а не эффектом.

И ВТОРОЙ ВОПРОС, РАДИ КОТОРОГО ЗАМЕР СТОИТ ДЕЛАТЬ ДВАЖДЫ: `mx.compile`
сам сливает цепочки поэлементных операций. Если хвост уже слит
компилятором, то ручной кернель экономит только редукции и запуски
вокруг них, и тогда его цена (ещё одна реализация той же математики,
закон 23) может не окупиться. Поэтому меряется И скомпилированный путь,
И сырой: разница между этими двумя ответами и есть ответ.

    python tests/bench_tail_ab.py [model.rwkvq] [раундов]
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


def swap_mb():
    env = dict(os.environ, LC_ALL="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True, env=env).stdout.replace("=", " ").split()
    return float(out[out.index("used") + 1].rstrip("M").replace(",", "."))


def main():
    qm.FUSE = True
    model = qm.QuantRWKV7(load_raw(PATH))
    prompt = mx.array(np.array([[1, 2, 3, 4]], dtype=np.int32))

    def bench(tail, compiled, n=30, warm=8):
        qm.FUSE_TAIL = tail
        fn = mx.compile(model.forward_stateful) if compiled \
            else model.forward_stateful
        st = model.init_state(1)
        logits, st = fn(prompt, st, True)
        tok = mx.argmax(logits[:, -1], axis=-1)
        for _ in range(warm):
            logits, st = fn(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(tok)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, st = fn(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(tok)
        mx.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    sw0 = swap_mb()
    for compiled in (True, False):
        # ЧЕРЕДОВАНИЕ: пара (off, on) внутри раунда, медиана по раундам,
        # рядом печатается разброс -- отношение защищено, абсолюты нет
        pairs = [(bench(False, compiled), bench(True, compiled))
                 for _ in range(ROUNDS)]
        off = float(np.median([p[0] for p in pairs]))
        on = float(np.median([p[1] for p in pairs]))
        spread = 100 * (max(p[0] for p in pairs) - min(p[0] for p in pairs)) / off
        tag = "mx.compile" if compiled else "сырой путь"
        print(f"{tag:12s}: хвост off {off:6.2f} мс | on {on:6.2f} мс | "
              f"выигрыш {off - on:+.2f} мс ({100*(off-on)/off:+.1f}%), "
              f"разброс off по раундам {spread:.1f}%")
        print("   пары:", " ".join(f"({a:.2f},{b:.2f})" for a, b in pairs))
    sw1 = swap_mb()
    print(f"своп: {sw0:.0f} -> {sw1:.0f} МБ (дельта {sw1-sw0:+.0f}; "
          f"ненулевая делает замер недействительным)")


if __name__ == "__main__":
    main()
