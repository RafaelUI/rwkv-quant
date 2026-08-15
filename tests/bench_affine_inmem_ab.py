"""A/B: наша раскладка в памяти против affine-транскрипции. ОДНА МОДЕЛЬ
НА ПРОЦЕСС, чередование ПРОЦЕССАМИ.

ПОЧЕМУ НЕ ПО ЗАКОНУ 1 (обе модели в одном процессе). Пробовал -- процесс
вырос до 7 ГБ и машина ушла в своп: `to_affine` освобождает наши буферы
только ПО УЧЁТУ ЖИВЫХ, а аллокатор MLX держит их в кеше, плюс транзиенты
декванта каждой матрицы. Своп во время замера делает его недействительным
безусловно (закон 11), поэтому здесь он бьёт закон 1. Расплата -- разброс
процессного чередования (5-18% против 0.7-1.3%, закон 24), и он
печатается рядом. Ожидаемые эффекты (префилл в разы, декод десятки
процентов) заведомо крупнее этого разброса -- иначе такой замер был бы
непригоден.

Пик снимается снаружи (`/usr/bin/time -l`, закон 22) и ВКЛЮЧАЕТ
транзиент перевода в affine -- это честно: сегодня перевод делается
после сборки модели. Стационарное потребление печатается обходом живых
буферов отдельно.

    /usr/bin/time -l python tests/bench_affine_inmem_ab.py base|a6|a8
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
from eval_affine_inmem import to_affine, live_mb, MODES  # noqa: E402

PATH = os.environ.get("RWKVQ_MODEL", "/tmp/reduction_sym_head8.rwkvq")
MODE = sys.argv[1] if len(sys.argv) > 1 else "base"
NSTEP = 25
T = 512
BURSTS = 3


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def main():
    sw0 = swap_mb()
    model = qm.QuantRWKV7(load_raw(PATH))
    if MODES[MODE]:
        bits, gs = MODES[MODE]
        to_affine(model, bits, gs)
        mx.clear_cache()          # иначе освобождённые наши буферы висят в кеше
    lm = live_mb(model)
    fn = mx.compile(model.forward_stateful)
    idx1 = mx.array(np.array([[187]], dtype=np.int32))
    idxP = mx.array(np.random.RandomState(7).randint(
        0, 65500, size=(1, T)).astype(np.int32))

    def decode(n=NSTEP):
        st = model.init_state(1)
        lg, st = fn(idx1, st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        for _ in range(5):
            lg, st = fn(tok[None], st)
            tok = mx.argmax(lg[:, -1], axis=-1); mx.eval(tok)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            lg, st = fn(tok[None], st)
            tok = mx.argmax(lg[:, -1], axis=-1); mx.eval(tok)
        mx.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    def prefill():
        st = model.init_state(1)
        lg, st = fn(idxP, st); mx.eval(lg); mx.synchronize()
        st = model.init_state(1)
        t0 = time.perf_counter()
        lg, st = fn(idxP, st); mx.eval(lg)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    # RWKVQ_NOPREFILL=1 -- прогон БЕЗ префилла: разность пиков с обычным
    # прогоном и есть то, сколько префилл добавляет поверх модели. Иначе
    # пик описывает конверсию (у affine) или деквант-транзиенты (у нас),
    # и сравнивать их между собой бессмысленно.
    noprefill = os.environ.get("RWKVQ_NOPREFILL") == "1"
    decode(3)
    if not noprefill:
        prefill()
    ds = [decode() for _ in range(BURSTS)]
    ps = [0.0] if noprefill else [prefill() for _ in range(BURSTS)]
    print(f"MODE={MODE} живые {lm:.1f} МБ | декод {np.median(ds):.2f} мс/ток "
          f"({1e3/np.median(ds):.1f} ток/с) | pp{T} {np.median(ps):.1f} мс "
          f"({T/np.median(ps)*1e3:.1f} ток/с) | своп {sw0:.0f}->{swap_mb():.0f}",
          flush=True)


if __name__ == "__main__":
    main()
