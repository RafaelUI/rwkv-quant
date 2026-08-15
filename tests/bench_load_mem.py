"""ПАМЯТЬ ЗАГРУЗКИ: только постройка модели из готового .rwkvq, без префилла.

ЗАЧЕМ ОТДЕЛЬНО ОТ bench_prefill_mem. Тот замер показал 6.22 ГБ пика, из
которых префилл добавляет 0.18 -- всё остальное это QuantRWKV7.__init__.
Здесь префилла нет вовсе, поэтому `peak memory footprint` описывает ровно
загрузку и ничего кроме.

МЕРИТЬ ТОЛЬКО СИСТЕМНЫМИ КОМАНДАМИ (законы 11 и 22): метрики MLX для
unified memory врут, RSS тоже (mmap считает страницы файла резидентными).

    /usr/bin/time -l python tests/bench_load_mem.py <model.rwkvq> [--prefill T]

Своп печатается на границе замера: рост свопа во время замера СКОРОСТИ
делает его недействительным; здесь мерится память, но знать всё равно надо.
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402


def swap_mb():
    """vm.swapusage через LC_ALL=C: локаль с запятичным разделителем иначе
    ломает разбор (у нас именно такая)."""
    env = dict(os.environ, LC_ALL="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True, env=env).stdout
    # "total = 2048.00M  used = 929.12M  free = ..." -- берём поле после used
    parts = out.replace("=", " ").split()
    try:
        i = parts.index("used")
        return float(parts[i + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def main():
    path = sys.argv[1]
    T = 0
    if "--prefill" in sys.argv:
        T = int(sys.argv[sys.argv.index("--prefill") + 1])

    if "--old" in sys.argv:
        # A/B в одинаковой оболочке: подменяем ТОЛЬКО путь построения
        # dense-параметров на замороженную копию прежнего кода из гейта.
        # Иначе «до» и «после» отличались бы не одной правкой, а версией
        # всего файла (в рабочем дереве лежит и работа прошлой сессии).
        import test_dense_load_parity as frozen
        import rwkv_quant.backends.metal.quant_model as qm
        qm._dense = frozen._old_dense
        print("ПРЕЖНИЙ путь _dense (замороженная копия)", flush=True)

    sw0 = swap_mb()
    t0 = time.time()
    ckpt = load_raw(path)
    t_read = time.time() - t0
    t0 = time.time()
    model = QuantRWKV7(ckpt)
    t_build = time.time() - t0
    del ckpt
    gc.collect()
    mx.clear_cache()

    print(f"{os.path.basename(path)}: файл {os.path.getsize(path)/1e6:.1f} МБ")
    print(f"load_raw {t_read:.2f} с, постройка {t_build:.1f} с")

    if T:
        rng = np.random.default_rng(0)
        prompt = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
        st = model.init_state(1)
        logits, st = model.forward_stateful(prompt, st, last_only=True)
        mx.eval(logits)
        print(f"префилл {T}: ок")

    sw1 = swap_mb()
    print(f"своп: {sw0:.0f} -> {sw1:.0f} МБ (дельта {sw1-sw0:+.0f})")
    print("пик смотреть в 'peak memory footprint' из /usr/bin/time -l")


if __name__ == "__main__":
    main()
