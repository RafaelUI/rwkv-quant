"""Префилл: скорость И ПАМЯТЬ. Один конфиг на процесс.

ЗАЧЕМ ОТДЕЛЬНО ОТ ДЕКОДА. Префилл идёт другим путём: при N >=
GEMM_MIN_BATCH_NB веса РАЗВОРАЧИВАЮТСЯ в плотный fp16 (`_dequant_w`) и
считаются обычным матмулом. Это даёт скорость, но создаёт транзиент
размером с матрицу на каждую проекцию -- на голове 1.5B это 268 МБ за
один вызов. Декодные замеры про это не говорят ничего: там путь GEMV и
транзиентов нет.

ПОЧЕМУ МОДЕЛЬ ГРУЗИТСЯ ИЗ .rwkvq, А НЕ КВАНТУЕТСЯ НА МЕСТЕ. Сборка из
.pth держит воркспейс грид-поиска и даёт пик 7.7 ГБ -- он перекрыл бы
всё, что мы хотим увидеть в префилле. Из файла загрузка стоит
десятые доли гигабайта, и `peak memory footprint` из /usr/bin/time -l
описывает тогда именно префилл.

    /usr/bin/time -l python tests/bench_prefill_mem.py <model.rwkvq> [T]
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
# CHUNK -- гипотеза про ленивость MLX: forward_stateful на 512 токенов
# строит граф целиком и считает в конце, поэтому промежуточные тензоры
# ВСЕХ слоёв могут жить одновременно. Разбиение промпта на куски с
# переносом состояния обрывает граф и должно ограничить пик, ничего не
# меняя в математике (state переносится ровно так же).
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def main():
    t0 = time.time()
    model = QuantRWKV7(load_raw(PATH))
    gc.collect()
    mx.clear_cache()
    print(f"{os.path.basename(PATH)}: файл "
          f"{os.path.getsize(PATH)/1e6:.1f} МБ, загрузка {time.time()-t0:.0f}s",
          flush=True)

    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))

    def prefill():
        st = model.init_state(1)
        if not CHUNK:
            # СКОМПИЛИРОВАННЫЙ путь (model.step обслуживает и префилл):
            # +35% скорости, см. докстринг QuantRWKV7.step. Пик памяти при
            # этом меняется, поэтому число из этого замера сравнивать
            # только с числами того же пути.
            logits, st = model.step(prompt, st, True)
            mx.eval(logits)
            return logits
        logits = None
        for a in range(0, T, CHUNK):
            logits, st = model.step(prompt[:, a:a + CHUNK], st, True)
            mx.eval(logits, *st)      # ОБРЫВ ГРАФА: считаем кусок целиком
        return logits

    logits = prefill()                   # прогрев компиляции
    del logits
    mx.clear_cache()

    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        logits = prefill()
        ts.append(time.perf_counter() - t0)
        del logits
    dt = float(np.median(ts))
    print(f"pp{T}{f' кусками по {CHUNK}' if CHUNK else ' одним куском'}: "
          f"{T/dt:.1f} ток/с  ({dt*1000:.1f} мс, "
          f"разброс {100*(max(ts)-min(ts))/dt:.1f}%)")
    print("пик смотреть в 'peak memory footprint' из /usr/bin/time -l")


if __name__ == "__main__":
    main()
