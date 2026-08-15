"""ПОТОЛОК ПОЛОСЫ ПАМЯТИ: сколько GPU РЕАЛЬНО вытягивает из DRAM.

ЗАЧЕМ. В проекте ходят три числа, и они меряют РАЗНОЕ:

  120 ГБ/с  -- НЕ ЗАМЕР ВООБЩЕ. Это паспортная арифметика шины:
               LPDDR5X-7500 x 128 бит = 7500 МТ/с x 16 байт. Столько
               способен передать контроллер при идеальном потоке, без
               refresh, без смены направления, без промахов страницы.
  95.7 ГБ/с -- закон 12: ОДИН замер потокового чтения fp16 средствами
               MLX (2 ГБ за 20.9 мс). Отсюда взялось «98% полосы» для
               нашего GEMV, то есть весь вывод «оптимизировать нечего».
  113.5 ГБ/с -- наш собственный GEMV на голове sym@6. Он БЫСТРЕЕ
               «достижимого», и это прямое указание, что 95.7 -- свойство
               замера, а не железа.

Пока потолок не измерен инструментом, который к нему стремится, «98% от
достижимого» есть утверждение о `mx.sum`, а не о машине. Разница
принципиальная: если потолок 95.7, декод почти на полке и работать надо
над трафиком; если он ближе к 110-115, у нас на столе лежат 15-20%.

ЧТО ЗДЕСЬ МЕРИТСЯ. Три инструмента на ОДНОМ буфере, чередованием в одном
процессе (закон 1), плюс развёртка по числу потоков -- ЧЕРЕДУЕМАЯ, потому
что свип это тоже замер скорости (закон 24):

  1. `mx.sum` -- ровно то, чем получены 95.7 (воспроизведение).
  2. Свой Metal-кернель чистого чтения: uint4-загрузки (16 байт на поток
     за итерацию), соседние потоки читают соседние адреса, аккумулятор
     целочисленный и уходит в выход -- иначе компилятор выбросит чтение.
  3. Копия (чтение + запись) -- для справки: она ограничена суммой
     трафика в обе стороны и показывает, во что обходится смена
     направления.

Буфер берётся заведомо больше системного кэша, иначе замерим SLC, а не
DRAM. Своп печатается на границе: рост во время замера скорости делает
его недействительным (закон 11).

    python tests/bench_membw.py [ГБ] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

GB = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
THREADS = (1 << 12, 1 << 14, 1 << 16)
TG = 256

_cache = {}


def swap_mb():
    env = dict(os.environ, LC_ALL="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True, env=env).stdout.replace("=", " ").split()
    return float(out[out.index("used") + 1].rstrip("M").replace(",", "."))


def read_kernel(nthreads, unroll=1):
    """Чистое чтение. Развёртка U даёт потоку U ПОДРЯД идущих uint4 за
    итерацию: внутри варпа адреса всё равно слитные, но каждый поток
    держит более длинный последовательный участок, а для DRAM важна
    локальность по страницам, а не только слитность по варпу."""
    key = ("r", nthreads, unroll)
    if key in _cache:
        return _cache[key]
    body = "\n".join(
        f"        s += p[i + {u}*NTH];" for u in range(unroll))
    src = f"""
    uint gid = thread_position_in_grid.x;
    device const uint4* p = (device const uint4*)src;
    uint4 s = uint4(0);
    for (uint i = gid; i + {unroll - 1}*NTH < N4; i += NTH*{unroll}) {{
{body}
    }}
    out[gid] = s.x + s.y + s.z + s.w;
"""
    k = mx.fast.metal_kernel(
        name=f"membw_read_{nthreads}_u{unroll}", input_names=["src"],
        output_names=["out"],
        header=f"constant uint NTH = {nthreads};\nconstant uint N4 = {N4};\n",
        source=src)
    _cache[key] = k
    return k


def copy_kernel(nthreads):
    if ("c", nthreads) in _cache:
        return _cache[("c", nthreads)]
    src = """
    uint gid = thread_position_in_grid.x;
    device const uint4* p = (device const uint4*)src;
    device uint4* q = (device uint4*)out;
    for (uint i = gid; i < N4; i += NTH) q[i] = p[i];
"""
    k = mx.fast.metal_kernel(
        name=f"membw_copy_{nthreads}", input_names=["src"],
        output_names=["out"],
        header=f"constant uint NTH = {nthreads};\nconstant uint N4 = {N4};\n",
        source=src)
    _cache[("c", nthreads)] = k
    return k


NBYTES = int(GB * (1 << 30))
NBYTES -= NBYTES % (16 * (1 << 20))          # кратно 16 МБ
N4 = NBYTES // 16
NHALF = NBYTES // 2


def timeit(fn, n=5):
    fn()                                      # прогрев
    mx.synchronize()
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    print(f"буфер {NBYTES/1e9:.2f} ГБ fp16 "
          f"(системный кэш M4 -- единицы МБ, так что это DRAM, не SLC)")
    x = mx.array(np.ones(NHALF, dtype=np.float16))
    mx.eval(x)
    sw0 = swap_mb()

    variants = [("mx.sum (закон 12)", lambda: mx.eval(mx.sum(x)))]
    for nt in THREADS:
        for u in (1, 4):
            variants.append((
                f"чтение, {nt} потоков x{u}",
                lambda nt=nt, u=u: mx.eval(read_kernel(nt, u)(
                    inputs=[x], grid=(nt, 1, 1), threadgroup=(TG, 1, 1),
                    output_shapes=[(nt,)], output_dtypes=[mx.uint32])[0])))
    nt = THREADS[1]
    variants.append((
        f"копия (чтение+запись), {nt}",
        lambda: mx.eval(copy_kernel(nt)(
            inputs=[x], grid=(nt, 1, 1), threadgroup=(TG, 1, 1),
            output_shapes=[(NHALF,)], output_dtypes=[mx.float16])[0])))

    res = {lab: [] for lab, _ in variants}
    for _ in range(ROUNDS):
        for lab, fn in variants:              # ЧЕРЕДОВАНИЕ (законы 1, 24)
            res[lab].append(timeit(fn))
    sw1 = swap_mb()

    print(f"\n{'инструмент':>30} | {'мс':>7} | {'ГБ/с':>7} | "
          f"{'% от 120':>8} | разброс")
    for lab, _ in variants:
        t = float(np.median(res[lab]))
        moved = NBYTES * (2 if lab.startswith("копия") else 1)
        bw = moved / t / 1e9
        spread = 100 * (max(res[lab]) - min(res[lab])) / t
        print(f"{lab:>30} | {t*1e3:7.2f} | {bw:7.1f} | {100*bw/120:7.1f}% | "
              f"{spread:.1f}%")
    print(f"\nсвоп: {sw0:.0f} -> {sw1:.0f} МБ (дельта {sw1-sw0:+.0f}; "
          f"ненулевая делает замер недействительным)")
    print("120 ГБ/с -- паспортная арифметика шины, а не чей-либо замер.")


if __name__ == "__main__":
    main()
