"""СКОЛЬКО СТОИТ ОБВЯЗКА ВОКРУГ НАШЕГО GEMV НА ДЕКОДЕ.

ЗАЧЕМ. Штатный mlx-affine быстрее нас на декоде в 1.10x (замер
`bench_molly_ab`), и разрыв раскладывается на две части: +3.7% трафика
(наша голова на восьми битах против их шести) и −6% эффективности по
полосе (74.6 против 79.4 ГБ/с). Второе -- кандидат на «взять их логику»:
у них слой это ОДИН `mx.quantized_matmul`, у нас питоновская обвязка --
reshape в 2D, отдельная операция `xbsum` на каждый вызов (при шести
битах), цикл по колонкам, конкатенация.

ТРИ ТОЧКИ:
  полный   -- `lin(x)` как есть;
  кернель  -- тот же launch, но `xbsum` посчитан ЗАРАНЕЕ и вне замера.
              Это ВЕРХНЯЯ оценка того, что дало бы перенесение xbsum
              внутрь кернеля: там эта работа никуда не денется, просто
              перестанет быть отдельным запуском и лишним проходом по x;
  нативный -- `mx.quantized_matmul` по affine-копии ТЕХ ЖЕ значений.

Меряется по ВСЕМ слоям за проход (рабочий набор настоящий, а не один
слой в кэше) и цепочкой на одну синхронизацию: при N=1 операция стоит
десятки микросекунд при поле синхронизации 0.238 мс, и поштучный замер
здесь шумит втрое (записано 15.08: 26 против 76 ГБ/с между прогонами).

    python tests/probe_gemv_wrapper.py [model.rwkvq] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.backends.metal import quant_linear_sym as qls  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
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
    sw0 = swap_mb()
    model = qm.QuantRWKV7(load_raw(PATH))
    groups = [("cmix key", [b.cmix.key for b in model.blocks]),
              ("proj r", [b.tmix.r_proj for b in model.blocks])]
    print(f"{'группа':>10} {'бит':>4} {'слоёв':>6} | {'полный':>9} "
          f"{'кернель':>9} {'нативный':>9} | {'обвязка':>8} {'кернель/нат':>11}")
    for name, lins in groups:
        lins = [l for l in lins if isinstance(l, qls.SymQuantLinear)]
        if not lins:
            continue
        l0 = lins[0]
        IN, OUT, bits = l0.in_features, l0.out_features, l0.bits
        xs = [mx.random.normal((1, IN)) for _ in lins]
        mx.eval(xs)
        # заранее: xbsum и конфигурация -- ровно то, что делает обвязка
        NSG, RS = qls._cfg(IN, OUT)
        n_tg = OUT // (NSG * RS)
        xb = [mx.sum(x.reshape(1, l0.NB, 16), axis=2) for x in xs]
        mx.eval(xb)
        if bits == 8:
            kern = qls._get_kernel_sym8(IN, OUT, NSG, RS, 1)
            inputs = lambda i: [xs[i], lins[i].qblk, lins[i].qs, lins[i].d]
        else:
            kern = qls._get_kernel_sym6(IN, OUT, NSG, RS, 1)
            inputs = lambda i: [xs[i], lins[i].qblk, lins[i].qs, lins[i].d, xb[i]]

        def bare():
            return [kern(inputs=inputs(i),
                         grid=(n_tg * NSG * 32, 1, 1),
                         threadgroup=(NSG * 32, 1, 1),
                         output_shapes=[(1, OUT)], output_dtypes=[mx.float32])[0]
                    for i in range(len(lins))]

        # битность нативного ОБЯЗАНА совпадать с нашей: при 6 битах
        # affine gs=64 стоит 6.5 бит/вес против наших 6.5625, при 8 --
        # 8.5 против 8.5625. Сравнивать 6-битное ядро с 8-битным значило
        # бы мерить разницу в байтах и называть её разницей ядер.
        qs = [mx.quantize(l._dequant_w(), group_size=64, bits=bits)
              for l in lins]
        mx.eval([t for tr in qs for t in tr])
        x16 = [x.astype(mx.float16) for x in xs]
        mx.eval(x16)

        def native():
            return [mx.quantized_matmul(x16[i], qs[i][0], scales=qs[i][1],
                                        biases=qs[i][2], transpose=True,
                                        group_size=64, bits=bits)
                    for i in range(len(lins))]

        def full():
            return [lins[i](xs[i]) for i in range(len(lins))]

        res = {k: [] for k in ("full", "bare", "nat")}
        for _ in range(ROUNDS):
            for k, fn in (("full", full), ("bare", bare), ("nat", native)):
                res[k].append(timeit(fn))
        f, b, n = (float(np.median(res[k])) for k in ("full", "bare", "nat"))
        print(f"{name:>10} {bits:>4} {len(lins):>6} | {f*1e3:6.3f} мс "
              f"{b*1e3:6.3f} мс {n*1e3:6.3f} мс | {f/b:7.2f}x {b/n:10.2f}x")
        del qs
    print(f"\nобвязка = полный/кернель (что можно снять переносом xbsum и "
          f"reshape внутрь);\nкернель/нат -- сравнение ядер на одних "
          f"значениях И одной битности.")
    print(f"своп: {sw0:.0f} -> {swap_mb():.0f} МБ")


if __name__ == "__main__":
    main()
