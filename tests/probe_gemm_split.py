"""ГДЕ ИМЕННО УХОДИТ ВРЕМЯ GEMM-ПРЕФИЛЛА: деквант, матмул или рейс через DRAM.

Проигрыш 2.2-2.5x нативному матмулу списывался на то, что мы
материализуем плотную матрицу. Но `_dequant_w` -- это не одна запись:
шестибитная ветка собирает коды ЦЕПОЧКОЙ MLX-операций (concatenate
нибблов, две битплоскости через сдвиги), и каждая рождает полноразмерный
промежуточный тензор. Прежде чем писать тайловый GEMM-кернель, надо
знать, сколько стоит КАЖДАЯ из трёх частей:

  t_dq   -- сам деквант (со всеми промежуточными);
  t_mm   -- матмул по уже готовой плотной матрице;
  t_all  -- полный вызов слоя.

Если t_dq >> t_mm, то первый шаг -- деквант ОДНИМ кернелем (логика
распаковки уже написана в GEMV), и тайловый GEMM можно отложить.

Плюс нарезка по строкам выхода: она не меняет трафик по весам, но
уменьшает транзиент и может удержать плотный кусок в системном кэше.

    python tests/probe_gemm_split.py [model.rwkvq] [N]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512


def timeit(fn, reps=5):
    mx.eval(fn()); mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn()); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    model = qm.QuantRWKV7(load_raw(PATH))
    tm, cm = model.blocks[1].tmix, model.blocks[1].cmix
    targets = [("cmix key [8192,2048] @6", cm.key),
               ("cmix value [2048,8192] @6", cm.value),
               ("proj r [2048,2048] @6", tm.r_proj),
               ("head [65536,2048] @8", model.head)]
    print(f"N={N}\n{'матрица':>26} | {'весь вызов':>10} {'деквант':>9} "
          f"{'матмул':>8} {'сумма':>7} | доля декванта")
    for name, lin in targets:
        x = mx.random.normal((N, lin.in_features)).astype(mx.float16)
        mx.eval(x)
        w = lin._dequant_w()
        mx.eval(w)
        t_mm = timeit(lambda: mx.matmul(x, w.T))
        del w
        t_dq = timeit(lambda: lin._dequant_w())
        t_all = timeit(lambda: lin(x))
        print(f"{name:>26} | {t_all*1e3:7.2f} мс {t_dq*1e3:6.2f} мс "
              f"{t_mm*1e3:5.2f} мс {(t_dq+t_mm)*1e3:4.2f} мс | "
              f"{t_dq/(t_dq+t_mm)*100:4.0f}%")
        del x

    # нарезка по строкам выхода: транзиент меньше, кусок может осесть в кэше
    print("\nНАРЕЗКА ПО СТРОКАМ ВЫХОДА (cmix key), кусок строк -> время вызова:")
    lin = cm.key
    x = mx.random.normal((N, lin.in_features)).astype(mx.float16)
    mx.eval(x)
    base = timeit(lambda: lin(x))
    OUT = lin.out_features

    def chunked(rows):
        def go():
            outs = []
            for i in range(0, OUT, rows):
                w = lin._dequant_w()[i:i + rows]
                outs.append(mx.matmul(x, w.T))
            return mx.concatenate(outs, axis=1)
        return go

    print(f"  целиком: {base*1e3:.2f} мс")
    for rows in (2048, 1024, 512):
        t = timeit(chunked(rows), reps=3)
        print(f"  по {rows:>5} строк: {t*1e3:7.2f} мс  {base/t:5.2f}x "
              f"(деквант тут пересчитывается целиком -- верхняя оценка цены "
              f"нарезки без порезанного декванта)")


if __name__ == "__main__":
    main()
