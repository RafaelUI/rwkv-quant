"""ГДЕ ИМЕННО МЫ ПРОИГРЫВАЕМ `mx.quantized_matmul`: кернель или обвязка?

ЧТО УЖЕ ИЗВЕСТНО И ЧТО НЕТ. `bench_molly_ab` показал, что штатно
квантованная модель быстрее нашей в 1.18 раза на декоде и в 1.39 на
префилле. Тот замер честно изолирует ЛИНЕЙНЫЙ СЛОЙ -- обе модели идут
через один и тот же наш forward, отличается только то, что стоит в
`_linear`. Но «линейный слой» -- это не одно и то же, что «GPU-кернель»:
у нас между питоном и Metal лежит обвязка, которой у встроенного примитива
нет вовсе.

  - при шести битах мы считаем `xbsum` ОТДЕЛЬНОЙ операцией на каждый
    вызов (`mx.sum` по блокам входа) -- это 144 лишних редукции за токен;
  - reshape'ы входа и выхода, цикл по колонкам при N > 1;
  - вызов через `mx.fast.metal_kernel`, а не через примитив, который
    компилятор MLX знает и может планировать иначе.

Разделяется это тремя точками на ОДНИХ И ТЕХ ЖЕ весах и формах:

  native  -- `mx.quantized_matmul` по affine-упаковке тех же значений;
  ours    -- наш `SymQuantLinear.__call__` целиком, как в модели;
  kernel  -- наш кернель напрямую, с ЗАРАНЕЕ посчитанным xbsum и без
             обвязки, то есть чистое время GPU-части.

Если kernel ~ native, то проигрывает обвязка, и чинится это питоном.
Если kernel << native, то проигрывает сам кернель, и тогда вопрос стоит
иначе -- нужен ли он вообще.

Веса берутся ИЗ НАСТОЯЩЕГО чекпоинта (закон 17), формы -- те, что реально
есть в 1.5B. Битность native берётся РАВНОЙ нашей на каждом тензоре,
иначе сравнивались бы разные объёмы трафика.

    python tests/bench_gemv_vs_native_ab.py [model.rwkvq] [раундов]
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.backends.metal.quant_linear_sym import (  # noqa: E402
    SymQuantLinear, _cfg, _get_kernel_sym6, _get_kernel_sym8)
from rwkv_quant.formats.reader import load_raw, dequantize_banded  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NS = (1, 128)
KEYS = ["blocks.0.att.receptance.weight", "blocks.3.ffn.key.weight",
        "blocks.3.ffn.value.weight", "head.weight"]
GS = 64


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


CHAIN_MAX = 16
SYNC_FLOOR_MS = 0.238      # пол синхронизации, замерен bench_step_decompose
CHAIN_TARGET_MS = 5.0      # столько работы копим на одну синхронизацию


def timeit(fn, reps=7):
    """Цепочка вызовов на одну синхронизацию, ДЛИНА ЦЕПОЧКИ АДАПТИВНАЯ.

    Зачем цепочка. Версия, синхронизировавшаяся на КАЖДОМ вызове, мерила
    пол синхронизации (0.238 мс), а не матвек: у [2048, 2048] при шести
    битах едет 3.4 МБ, то есть ~0.035 мс работы при 100 ГБ/с. Она
    показывала 0.29 мс у обеих реализаций и «паритет», которого не могла
    не показать -- сравнивались две синхронизации.

    ЗАЧЕМ АДАПТИВНАЯ. Фиксированная цепочка из 16 вызовов УБИЛА МАШИНУ на
    N=512: MLX ленив, поэтому шестнадцать неотевалуированных вызовов
    держат шестнадцать промежуточных результатов ЖИВЫМИ одновременно, а
    при N >= GEMM_MIN_BATCH_NB наш путь идёт через `_dequant_w`, то есть
    материализует ПЛОТНУЮ матрицу целиком: на голове 268 МБ fp16 плюс
    выход 134 МБ, шестнадцать раз -- 6.4 ГБ. Своп вырос на 1.5 ГБ, замер
    убит вместе с машиной.

    Это тот же сорт ошибки, что закон 18 (гейт, державший эталон и
    пересчёт разом): приём, который чинит ОДИН режим замера, в другом
    режиме стоит памяти, линейной по длине цепочки. Поэтому длина
    выбирается ПО ИЗМЕРЕННОМУ времени одного вызова: копим примерно
    CHAIN_TARGET_MS работы и не больше CHAIN_MAX вызовов. Когда один
    вызов стоит 13-23 мс, цепочка вырождается в единицу -- и правильно,
    амортизировать там нечего (пол 0.238 мс -- это 1-2%).

    ОБОРОТНАЯ СТОРОНА, которая остаётся: цепочка держит матрицу горячей в
    системном кэше, если она туда влезает. Поэтому АБСОЛЮТНЫЕ ГБ/с здесь
    завышены и как оценку полосы их брать нельзя; честно только
    ОТНОШЕНИЕ -- кэш помогает обеим реализациям одинаково."""
    mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    t1_ms = (time.perf_counter() - t0) * 1e3
    chain = int(min(CHAIN_MAX, max(1, CHAIN_TARGET_MS // max(t1_ms, 1e-3))))

    mx.eval([fn() for _ in range(chain)])
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(chain)]
        mx.eval(outs)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / chain)
    del outs
    return float(np.median(ts))


def main():
    ckpt = load_raw(PATH)
    sw0 = swap_mb()
    print(f"своп на старте {sw0:.0f} МБ; рост во время замера делает его "
          f"недействительным (закон 11)", flush=True)
    print(f"{'тензор':>34} {'N':>4} | {'native':>9} {'ours':>9} "
          f"{'kernel':>9} | {'ours/nat':>8} {'kern/nat':>8} | ГБ/с nat / kern")

    for key in KEYS:
        qt = ckpt.tensors[key]
        if getattr(qt, "gw_mode", "") != "sym":
            print(f"{key}: не sym, пропуск"); continue
        lin = SymQuantLinear(qt)
        OUT, IN = lin.out_features, lin.in_features
        bits = lin.bits

        # те же значения в affine-упаковке, битность РАВНАЯ нашей
        w = mx.array(dequantize_banded(qt, __import__("torch").float16).numpy())
        Wq, Sc, Bi = mx.quantize(w, group_size=GS, bits=bits)
        mx.eval(Wq, Sc, Bi)
        del w
        gc.collect()

        # бит/вес: у нас bits + 8/16 + 16/(16*16); у affine bits + 2*16/GS
        mb_ours = OUT * IN * (bits + 0.5 + 16 / 256) / 8 / 1e6
        mb_nat = OUT * IN * (bits + 32 / GS) / 8 / 1e6

        NSG, RS = _cfg(IN, OUT)
        n_tg = OUT // (NSG * RS)

        for N in NS:
            x = mx.array(np.random.randn(N, IN).astype(np.float32))
            mx.eval(x)
            xh = x.astype(mx.float16)
            xb = (mx.sum(x.reshape(N, lin.NB, 16), axis=2)
                  if bits == 6 else None)
            mx.eval(xh, *( [xb] if xb is not None else []))

            def native():
                return mx.quantized_matmul(xh, Wq, scales=Sc, biases=Bi,
                                           transpose=True, group_size=GS,
                                           bits=bits)

            def ours():
                return lin(x)

            def kernel():
                # ЧИСТАЯ GPU-часть НАШЕГО GEMV: xbsum посчитан заранее,
                # reshape'ов нет. ОСМЫСЛЕН ТОЛЬКО ПРИ МАЛЫХ N: при N >= 128
                # реальный путь идёт не сюда, а через _dequant_w + матмул,
                # и гнать GEMV-кернель с NN=128 значит мерить то, чего в
                # модели не происходит.
                if bits == 8:
                    k = _get_kernel_sym8(IN, OUT, NSG, RS, N)
                    inp = [x, lin.qblk, lin.qs, lin.d]
                else:
                    k = _get_kernel_sym6(IN, OUT, NSG, RS, N)
                    inp = [x, lin.qblk, lin.qs, lin.d, xb]
                return k(inputs=inp, grid=(n_tg * NSG * 32, 1, 1),
                         threadgroup=(NSG * 32, 1, 1),
                         output_shapes=[(N, OUT)],
                         output_dtypes=[mx.float32])[0]

            cases = [("native", native), ("ours", ours)]
            if N < 128:                             # см. примечание в kernel()
                cases.append(("kernel", kernel))
            got = {}
            for _ in range(ROUNDS):                 # ЧЕРЕДОВАНИЕ (закон 1)
                for lab, fn in cases:
                    got.setdefault(lab, []).append(timeit(fn))
            t = {k2: float(np.median(v)) for k2, v in got.items()}
            nat, our = t["native"], t["ours"]
            kt = t.get("kernel")
            c_k = f"{kt*1e3:8.3f}м" if kt else "       --"
            c_kr = f"{kt/nat:7.2f}x" if kt else "      --"
            c_kb = f"{mb_ours/kt/1e3:5.0f}" if kt else "   --"
            print(f"{key[-32:]:>34} {N:>4} | "
                  f"{nat*1e3:8.3f}м {our*1e3:8.3f}м {c_k:>9} | "
                  f"{our/nat:7.2f}x {c_kr:>8} | "
                  f"{mb_nat/nat/1e3:5.0f} / {c_kb:>5}", flush=True)
            sw = swap_mb()
            if sw > sw0 + 32:
                print(f"  СВОП ВЫРОС {sw0:.0f} -> {sw:.0f} МБ -- строки "
                      f"ниже недействительны, замер прерван")
                sys.exit(1)
            del x, xh, xb
            gc.collect()
            mx.clear_cache()
        del lin, Wq, Sc, Bi
        gc.collect()
        mx.clear_cache()

    print(f"\nсвоп: {sw0:.0f} -> {swap_mb():.0f} МБ")
    print("ours/nat > kernel/nat  =>  проигрывает ОБВЯЗКА (питон, xbsum, "
          "reshape).\nkernel/nat > 1        =>  проигрывает сам КЕРНЕЛЬ.")


if __name__ == "__main__":
    main()
