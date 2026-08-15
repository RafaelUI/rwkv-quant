"""ЧТО ДАЁТ И ЧТО СТОИТ `mx.compile`: декод, ПРЕФИЛЛ и цена первого вызова.

ТРИ РАЗНЫХ ВОПРОСА, КОТОРЫЕ ПОСТОЯННО СЛИПАЮТСЯ В ОДИН.

1. СКОЛЬКО КОМПИЛЯЦИЯ ДАЁТ НА УСТАНОВИВШЕМСЯ ПУТИ. Для декода известно
   (+13%, и это записано в докстринге `QuantRWKV7.step`). Для ПРЕФИЛЛА
   не мерилось никогда: `bench_prefill_mem` и `bench_decode_vs_gguf`
   зовут сырой `forward_stateful`, причём во втором рядом стоит
   комментарий «прогрев mx.compile» -- то есть намерение было, а вызова
   нет. Значит записанные 348-354 ток/с префилла сняты БЕЗ компиляции, и
   разрыв с llama.cpp вдвое может быть отчасти этим.

2. СКОЛЬКО КОМПИЛЯЦИЯ СТОИТ ОДИН РАЗ. Трассировка плюс сборка
   Metal-библиотек -- это задержка ПЕРВОГО токена, которую пользователь
   чата видит целиком. Меряется как разница первого вызова и
   установившегося.

3. СКОЛЬКО РАЗ ЭТО ПРОИСХОДИТ. Кеш `mx.compile` ключуется ФОРМАМИ, то
   есть каждая новая длина промпта -- новая трассировка. Для чата, где
   длины произвольные, это не «один раз при старте», а «каждый раз».
   Проверяется прямо: скомпилировать, прогнать T=512, затем T=511 и
   посмотреть, вернулась ли цена первого вызова.

Всё, что сравнивается по скорости, чередуется в одном процессе (закон 1),
своп фиксируется на границе (закон 11).

    python tests/bench_compile_ab.py [model.rwkvq] [T] [раундов]
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
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def swap_mb():
    env = dict(os.environ, LC_ALL="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True, env=env).stdout.replace("=", " ").split()
    return float(out[out.index("used") + 1].rstrip("M").replace(",", "."))


def main():
    model = qm.QuantRWKV7(load_raw(PATH))
    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
    prompt2 = prompt[:, :T - 1]                  # другая форма -- другой кеш
    tok1 = mx.array(np.array([[1]], dtype=np.int32))

    def prefill(fn, p):
        st = model.init_state(1)
        logits, st = fn(p, st, True)
        mx.eval(logits)
        return logits

    def decode(fn, n=24):
        st = model.init_state(1)
        logits, st = fn(prompt, st, True)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok, *st)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, st = fn(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(tok)
        mx.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    def timed(f):
        mx.synchronize()
        t0 = time.perf_counter()
        r = f()
        mx.eval(r) if isinstance(r, mx.array) else None
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    raw = model.forward_stateful
    sw0 = swap_mb()

    # --- 2 и 3: цена первого вызова и её возврат на новой форме ----------
    print("=== цена компиляции (первый вызов против установившегося) ===")
    prefill(raw, prompt)                      # собрать наши Metal-кернели
    mx.synchronize()
    comp = mx.compile(raw)
    t_first = timed(lambda: prefill(comp, prompt))
    t_warm = min(timed(lambda: prefill(comp, prompt)) for _ in range(3))
    print(f"префилл T={T}: первый вызов {t_first:7.1f} мс, "
          f"установившийся {t_warm:7.1f} мс -> компиляция "
          f"{t_first - t_warm:6.1f} мс")
    t_first2 = timed(lambda: prefill(comp, prompt2))
    t_warm2 = min(timed(lambda: prefill(comp, prompt2)) for _ in range(3))
    print(f"префилл T={T-1} (ДРУГАЯ форма, тот же compiled): "
          f"первый {t_first2:7.1f} мс, установившийся {t_warm2:7.1f} мс "
          f"-> компиляция {t_first2 - t_warm2:6.1f} мс")
    print("   ^ если вторая цена не нулевая, кеш ключуется формой и в чате "
          "с произвольными длинами платится КАЖДЫЙ раз")

    comp_d = mx.compile(raw)
    st = model.init_state(1)
    t_first_d = timed(lambda: comp_d(tok1, st))
    t_warm_d = min(timed(lambda: comp_d(tok1, st)) for _ in range(5))
    print(f"декод T=1:   первый вызов {t_first_d:7.1f} мс, "
          f"установившийся {t_warm_d:7.1f} мс -> компиляция "
          f"{t_first_d - t_warm_d:6.1f} мс")

    # --- 1: установившаяся скорость, чередованием -----------------------
    print(f"\n=== установившаяся скорость, чередование, {ROUNDS} раундов ===")
    for _ in range(2):
        prefill(raw, prompt); prefill(comp, prompt)

    pre = {"сырой": [], "compiled": []}
    for _ in range(ROUNDS):
        pre["сырой"].append(timed(lambda: prefill(raw, prompt)))
        pre["compiled"].append(timed(lambda: prefill(comp, prompt)))
    a = float(np.median(pre["сырой"])); b = float(np.median(pre["compiled"]))
    sa = 100 * (max(pre["сырой"]) - min(pre["сырой"])) / a
    sb = 100 * (max(pre["compiled"]) - min(pre["compiled"])) / b
    print(f"префилл pp{T}: сырой {a:7.1f} мс ({T/a*1e3:6.1f} ток/с, "
          f"разброс {sa:.1f}%) | compiled {b:7.1f} мс ({T/b*1e3:6.1f} ток/с, "
          f"разброс {sb:.1f}%) | выигрыш {100*(a-b)/a:+.1f}%")

    dec = {"сырой": [], "compiled": []}
    for _ in range(ROUNDS):
        dec["сырой"].append(decode(raw))
        dec["compiled"].append(decode(comp_d))
    a = float(np.median(dec["сырой"])); b = float(np.median(dec["compiled"]))
    print(f"декод:       сырой {a:7.2f} мс/ток ({1000/a:5.1f} ток/с) | "
          f"compiled {b:7.2f} мс/ток ({1000/b:5.1f} ток/с) | "
          f"выигрыш {100*(a-b)/a:+.1f}%")

    sw1 = swap_mb()
    print(f"\nсвоп: {sw0:.0f} -> {sw1:.0f} МБ (дельта {sw1-sw0:+.0f})")


if __name__ == "__main__":
    main()
