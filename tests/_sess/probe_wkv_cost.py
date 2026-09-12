"""Сколько из не-GEMV части шага стоит само ядро WKV.

Не-GEMV пул на 09.09 -- 3.06 мс (23% шага) при собственном потолке по памяти
0.56 мс. Первый подозреваемый -- wkv7_infer: сетка (B*H, D) = (32, 64), то есть
32 группы по 64 потока, работа крошечная, а вызовов 24 и они зависимы по state.
Считается и трафик состояния: 24 x (чтение + запись) [1,H,S,S] fp32 -- он в
900.9 МБ/ток НЕ входит, там только веса.
"""
import os
import sys
import time

sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-metal")

import mlx.core as mx
import numpy as np
from rwkv_metal.kernel.wkv7 import wkv7_infer

H, S, N = 32, 64, 24
ROUNDS, REPS, WARM = 21, 21, 30


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
    rng = np.random.default_rng(0)
    def t4():
        return mx.array(rng.standard_normal((1, 1, H, S)).astype(np.float32))
    args = [[t4() for _ in range(6)] for _ in range(N)]
    hs = [mx.array(rng.standard_normal((1, H, S, S)).astype(np.float32) * 0.01)
          for _ in range(N)]
    w0 = [mx.array((-0.5 - 0.5 * np.abs(rng.standard_normal((1, 1, H, S)))).astype(np.float32))
          for _ in range(N)]
    for a in args:
        mx.eval(a)
    mx.eval(hs, w0)

    def chained():
        h = hs[0]
        out = None
        for i in range(N):
            r, _, k, v, a, b = args[i]
            out, h = wkv7_infer(r, w0[i], k, v, a, b, h)
        return [out, h]

    def independent():
        res = []
        for i in range(N):
            r, _, k, v, a, b = args[i]
            res.append(wkv7_infer(r, w0[i], k, v, a, b, hs[i])[0])
        return res

    def one():
        r, _, k, v, a, b = args[0]
        return list(wkv7_infer(r, w0[0], k, v, a, b, hs[0]))

    x = mx.array(np.zeros((16,), dtype=np.float32))
    cases = {"пол eval": lambda: x + 1.0, "wkv x1": one,
             "wkv x24 цепочкой": chained, "wkv x24 независимо": independent}
    acc = {k: [] for k in cases}
    for _ in range(ROUNDS):
        for k, fn in cases.items():
            acc[k].append(bench(fn))
    med = {k: float(np.median(v)) for k, v in acc.items()}
    print("%-20s %8s %8s" % ("что", "мс", "разброс"), flush=True)
    for k in cases:
        sp = 100 * (max(acc[k]) - min(acc[k])) / med[k]
        print("%-20s %8.3f %7.1f%%" % (k, med[k], sp), flush=True)
    floor = med["пол eval"]
    ch = med["wkv x24 цепочкой"] - floor
    ind = med["wkv x24 независимо"] - floor
    st_mb = 2 * N * H * S * S * 4 / 1e6
    print("", flush=True)
    print("WKV на шаг (цепочкой, за вычетом пола): %.3f мс" % ch, flush=True)
    print("то же независимо:                       %.3f мс" % ind, flush=True)
    print("цена зависимости:                       %.3f мс = %.1f мкс на вызов"
          % (ch - ind, (ch - ind) * 1000 / N), flush=True)
    print("трафик состояния (чтение+запись, 24 слоя): %.1f МБ, дно по памяти %.3f мс"
          % (st_mb, st_mb / 104), flush=True)


if __name__ == "__main__":
    main()
