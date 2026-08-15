"""ПОТОЛОК АРИФМЕТИКИ: сколько ФЛОПов машина реально отдаёт на матмуле.

ЭТО АНАЛОГ `bench_membw` ДЛЯ ПРЕФИЛЛА. На декоде пол считается от полосы
памяти (трафик за токен / 104 ГБ/с), и это верно, потому что веса
читаются на КАЖДЫЙ токен. На префилле они читаются ОДИН РАЗ НА ВЫЗОВ:
1149 МБ за pp512 -- это 11 мс из измеренных 753, то есть 1.5% времени.
Значит пол префилла считается от ФЛОПов:

    пол = 2 * P * T / пик  +  WKV-скан  +  поэлементное

где P -- параметры, участвующие в матмулах (emb не считается, это
gather; голова считается): 1.393 млрд на 1.5B, 2.779 на 2.9B.

ПАСПОРТНЫЕ ~4.3 ТФЛОПС ДЛЯ 10-ЯДЕРНОГО GPU M4 ИМЕЮТ ТОТ ЖЕ СТАТУС, ЧТО
120 ГБ/с У ШИНЫ: это арифметика, а не замер (закон 12 появился ровно
после того, как выяснилось, что на 120 стояли все выводы). Поэтому здесь
меряется практический потолок -- то, что выжимает `mx.matmul`.

Меряются ТРИ семейства, и разница между ними и есть ответ:
  1. квадратные -- самый дружелюбный случай, верхняя оценка;
  2. НАШИ формы при T=512 (M=512, K=2048, N=out) -- то, что реально
     считает префилл;
  3. те же формы через `mx.quantized_matmul` -- сколько отдаёт нативный
     квантованный путь.

Чередование внутри раунда (закон 24), медиана и разброс, своп на границе.
Гонять на ОСТЫВШЕЙ машине: потолок -- величина абсолютная, а декод
троттлит на 30-40% за минуту (закон 25).

    python tests/bench_gemm_peak.py [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
REPS = 3


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def timeit(fn):
    mx.eval(fn()); mx.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(fn()); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    cases = []

    # 1. квадратные fp16
    for n in (2048, 4096, 8192):
        a = mx.random.normal((n, n)).astype(mx.float16)
        b = mx.random.normal((n, n)).astype(mx.float16)
        mx.eval(a, b)
        cases.append((f"квадрат {n}x{n}x{n} fp16", 2 * n ** 3,
                      (lambda a=a, b=b: lambda: a @ b)()))

    # 2. наши формы при T=512
    T = 512
    ours = [("proj [2048,2048]", 2048, 2048),
            ("cmix key [8192,2048]", 2048, 8192),
            ("cmix value [2048,8192]", 8192, 2048),
            ("head [65536,2048]", 2048, 65536)]
    for name, K, N in ours:
        x = mx.random.normal((T, K)).astype(mx.float16)
        w = mx.random.normal((N, K)).astype(mx.float16)
        mx.eval(x, w)
        cases.append((f"наш {name} T=512 fp16", 2 * T * K * N,
                      (lambda x=x, w=w: lambda: x @ w.T)()))
        q = mx.quantize(w, group_size=64, bits=8)
        mx.eval(q)
        cases.append((f"  он же quantized_matmul@8", 2 * T * K * N,
                      (lambda x=x, q=q: lambda: mx.quantized_matmul(
                          x, q[0], scales=q[1], biases=q[2], transpose=True,
                          group_size=64, bits=8))()))

    sw0 = swap_mb()
    for _, _, fn in cases:
        mx.eval(fn())
    mx.synchronize()

    res = {i: [] for i in range(len(cases))}
    for r in range(ROUNDS):
        for i, (_, _, fn) in enumerate(cases):
            res[i].append(timeit(fn))
    sw1 = swap_mb()

    print(f"{'случай':>36} | {'мс':>8} {'разброс':>8} | {'ТФЛОП/с':>8}")
    print("-" * 72)
    best = 0.0
    for i, (name, fl, _) in enumerate(cases):
        ts = np.array(res[i])
        med = float(np.median(ts))
        tf = fl / med / 1e12
        best = max(best, tf)
        print(f"{name:>36} | {med*1e3:6.2f} мс {(ts.max()-ts.min())/med*100:6.1f}% "
              f"| {tf:8.2f}")
    print(f"\nлучший результат: {best:.2f} ТФЛОП/с")
    for tag, P, meas in (("1.5B", 1.393e9, 0.753), ("2.9B", 2.779e9, 1.440)):
        fl = 2 * P * 512
        print(f"  {tag}: pp512 = {fl/1e12:.3f} ТФЛОП, измерено {meas*1e3:.0f} мс "
              f"-> {fl/1e12/meas:.2f} ТФЛОП/с = {fl/1e12/meas/best*100:.0f}% "
              f"от лучшего; пол по арифметике {fl/1e12/best*1e3:.0f} мс")
    print(f"своп: {sw0:.0f} -> {sw1:.0f} МБ "
          f"({'ОК' if sw1 - sw0 < 1 else 'ЗАМЕР НЕДЕЙСТВИТЕЛЕН'})")


if __name__ == "__main__":
    main()
