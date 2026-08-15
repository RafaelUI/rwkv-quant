"""ЧТО ИМЕННО ДЕЛАЕТ LoRA-ВЕТКИ МЕДЛЕННЫМИ (46.6 ГБ/с против 85-93).

ТРИ ГИПОТЕЗЫ, КОТОРЫЕ НАДО РАЗДЕЛИТЬ.

  H1 ЗАПУСКИ. Восемь матмулов на слой, 192 за токен, каждый по 0.4-2 МБ.
     Диспетчеризация не амортизируется.
  H2 РАСКЛАДКА В ПАМЯТИ. `_mm(x, w)` считает `x @ w.T`, а сами веса
     получены как `_dense(...).T` -- то есть в игре два транспонирования.
     Если хоть одно материализуется копией на КАЖДОМ вызове, это объясняет
     половину полосы без всяких запусков.
  H3 ФОРМЫ. Ранги 96/96/64/256 против 2048: матмул [1, D] x [D, 96] мал
     настолько, что не насыщает ни полосу, ни занятость, и тут не
     поможет ничего, кроме укрупнения.

ПОЧЕМУ ПРЕЖНЯЯ ЦИФРА 2.15 мс ЗАНИЖЕНА. Аблация в `profile_components`
подменяла ранги на 1, а матмулы ОСТАВЛЯЛА -- все 192 запуска происходили в
обеих ветках и сократились в разности. То есть 2.15 мс -- это цена лишних
БАЙТОВ, а цена запусков в неё не вошла вовсе. Полная цена меряется здесь
подменой самого `_mm` на выдачу нулей нужной формы.

    python tests/probe_lora_cost.py [model.rwkvq] [раундов]
"""
import os
import subprocess
import sys
import time  # noqa: F401  (используется в timeit)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CHAIN = 32


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def timeit(fn, reps=7):
    """Цепочка на одну синхронизацию: операции тут по десятку микросекунд
    при поле синхронизации 0.238 мс. Выходы мелкие, память не растёт."""
    mx.eval([fn() for _ in range(CHAIN)])
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(CHAIN)]
        mx.eval(outs)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return float(np.median(ts))


def main():
    model = qm.QuantRWKV7(load_raw(PATH))
    tm = model.blocks[1].tmix
    D = model.n_embd
    sw0 = swap_mb()
    x = mx.array(np.random.randn(1, D).astype(np.float32))
    mx.eval(x)

    branches = [("w_lora", tm.w_lora_A, tm.w_lora_B_w),
                ("a_lora", tm.a_lora_A, tm.a_lora_B_w),
                ("v_lora", tm.v_lora_A, tm.v_lora_B_w),
                ("g_lora", tm.g_lora_A, tm.g_lora_B_w)]

    # --- H2: раскладка. Сравниваем `_mm` как есть с вариантом, где
    #     транспонированная матрица посчитана ЗАРАНЕЕ и непрерывна.
    print(f"{'ветка':>8} {'форма A':>14} | {'_mm как есть':>13} "
          f"{'A.T заранее':>13} | {'выигрыш':>8} | ГБ/с как есть")
    tot_asis = tot_pre = 0.0
    for name, A, B in branches:
        if A is None:
            continue
        At = mx.contiguous(A.T)              # [D, rank], непрерывная
        mx.eval(At)
        mb = A.size * 2 / 1e6
        t_asis = float(np.median([timeit(lambda: qm._mm(x, A))
                                  for _ in range(ROUNDS)]))
        t_pre = float(np.median([timeit(lambda: x.astype(mx.float16) @ At)
                                 for _ in range(ROUNDS)]))
        tot_asis += t_asis
        tot_pre += t_pre
        print(f"{name:>8} {str(A.shape):>14} | {t_asis*1e6:10.1f} мкс "
              f"{t_pre*1e6:10.1f} мкс | {t_asis/t_pre:7.2f}x | "
              f"{mb/t_asis/1e3:5.1f}")
    print(f"{'сумма вниз':>8} {'':>14} | {tot_asis*1e6:10.1f} мкс "
          f"{tot_pre*1e6:10.1f} мкс | {tot_asis/tot_pre:7.2f}x")

    # --- H1/H3: КОНКАТЕНАЦИЯ down-проекций в ОДИН матмул.
    #     Ранги 96+96+64+256 = 512, склейка идёт по выходной оси, поэтому
    #     паддинга нет вовсе и результат бит-в-бит тот же по каждой ветке.
    As = [A for _, A, _ in branches if A is not None]
    Acat = mx.contiguous(mx.concatenate(As, axis=0).T)   # [D, 512]
    mx.eval(Acat)
    t_cat = float(np.median([timeit(lambda: x.astype(mx.float16) @ Acat)
                             for _ in range(ROUNDS)]))
    mb_cat = sum(A.size for A in As) * 2 / 1e6
    print(f"\nЧЕТЫРЕ down-проекции ОДНИМ матмулом [{D}, "
          f"{sum(A.shape[0] for A in As)}]:")
    print(f"  порознь {tot_asis*1e6:8.1f} мкс | одним {t_cat*1e6:8.1f} мкс | "
          f"выигрыш {tot_asis/t_cat:.2f}x | "
          f"{mb_cat/t_cat/1e3:.1f} против {mb_cat/tot_asis/1e3:.1f} ГБ/с")
    print("  склейка по ВЫХОДНОЙ оси: паддинга нет, каждая ветка получает "
          "свой срез, значения те же")

    # --- потолок: та же склейка, но через НАТИВНЫЙ квантованный матмул.
    #     Это не гипотетика: LoRA в .rwkvq уже лежит в asym gw64, то есть
    #     w = q*scale + min по группам 64 -- РОВНО контейнер mlx-affine при
    #     group_size=64. Репак туда точный, значения не меняются, и по ppl
    #     эта группа стоит +0.006% (замер 12.08), то есть ничего.
    Wcat = mx.contiguous(mx.concatenate(As, axis=0))      # [512, D]
    mx.eval(Wcat)
    print("\nТА ЖЕ СКЛЕЙКА ЧЕРЕЗ mx.quantized_matmul (трафик реально "
          "прочитанных байт):")
    for bits in (8, 6):
        Wq, Sc, Bi = mx.quantize(Wcat, group_size=64, bits=bits)
        mx.eval(Wq, Sc, Bi)
        t_q = float(np.median([
            timeit(lambda: mx.quantized_matmul(
                x.astype(mx.float16), Wq, scales=Sc, biases=Bi,
                transpose=True, group_size=64, bits=bits))
            for _ in range(ROUNDS)]))
        mb_q = Wcat.size * (bits + 32 / 64) / 8 / 1e6
        print(f"  {bits} бит: {t_q*1e6:7.1f} мкс | {mb_q/t_q/1e3:5.1f} ГБ/с | "
              f"против порознь-fp16 {tot_asis/t_q:.2f}x, "
              f"против склейки-fp16 {t_cat/t_q:.2f}x")
        del Wq, Sc, Bi

    # --- полная цена LoRA-матмулов в шаге: подменяем сам _mm
    def bench_step(fn, n=25, warm=8):
        st = model.init_state(1)
        idx = mx.array(np.array([[123]], dtype=np.int32))
        logits, st = fn(idx, st)
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

    _mm_orig = qm._mm
    full = mx.compile(model.forward_stateful)
    deltas, fulls = [], []
    for _ in range(ROUNDS):
        tA = bench_step(full)
        qm._mm = lambda xx, ww: mx.zeros(xx.shape[:-1] + (ww.shape[0],),
                                         dtype=xx.dtype)
        abl = mx.compile(model.forward_stateful)
        tB = bench_step(abl)
        qm._mm = _mm_orig
        del abl
        fulls.append(tA)
        deltas.append(tA - tB)
    d = float(np.median(deltas))
    f = float(np.median(fulls))
    print(f"\nПОЛНАЯ цена LoRA-матмулов (_mm -> нули, запуски тоже уходят):"
          f" {d:.2f} мс из {f:.2f} ({100*d/f:.1f}% шага)")
    print(f"  для сравнения, аблация рангом 1 (запуски ОСТАЮТСЯ) давала "
          f"2.15 мс -- разность и есть цена ЗАПУСКОВ")
    print(f"\nсвоп: {sw0:.0f} -> {swap_mb():.0f} МБ")


if __name__ == "__main__":
    main()
