"""
Цена torch-free декванта (шаг 3 плана формата). Замер, из-за которого
reader.py оставлен на torch: раскладку унифицировали, арифметику -- нет.

Закон 1: чередование в одном процессе, безфановый дрейф иначе даёт до
1.8x на «том же» замере. Меряется суммарное время декванта набора
тензоров реального чекпоинта: codec (numpy, однопоточный) против
reader (torch, по ядрам), R раундов по очереди, печатается медиана.

    python tests/bench_codec_dequant_ab.py <model.rwkvq> [R] [N_TENSORS]
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from rwkv_quant.formats.reader import load_raw, _dequantize_one  # noqa: E402
from test_codec_parity import (codec_dequantize_one,  # noqa: E402
                               _ref_dequantize_one)


def run(fn, tensors):
    t0 = time.perf_counter()
    for qt in tensors:
        w = fn(qt)
        del w
    return (time.perf_counter() - t0) * 1e3


def main():
    path = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    ckpt = load_raw(path)
    # берём подряд идущий срез: он содержит полный слой всех раскладок
    tensors = [qt for qt in ckpt.tensors.values() if qt.bits < 16][:N]
    kinds = {}
    for qt in tensors:
        k = qt.gw_mode or "rtn"
        kinds[k] = kinds.get(k, 0) + 1
    n_el = sum(int(torch.tensor(qt.shape).prod()) for qt in tensors)
    print(f"{len(tensors)} тензоров ({kinds}), {n_el / 1e6:.1f}M элементов, "
          f"{R} раундов")

    # три пути: полностью numpy / сегодняшний reader (numpy-распаковка +
    # torch-арифметика) / полностью torch, как было до рефакторинга
    ms = {"codec/numpy": [], "reader": [], "torch-only": []}
    for r in range(R):
        ms["codec/numpy"].append(run(codec_dequantize_one, tensors))
        ms["reader"].append(run(_dequantize_one, tensors))
        ms["torch-only"].append(run(_ref_dequantize_one, tensors))
        print(f"  раунд {r}: " + "   ".join(
            f"{k} {v[-1]:7.1f}" for k, v in ms.items()))

    med = {k: statistics.median(v) for k, v in ms.items()}
    base = med["torch-only"]
    print()
    for k, v in med.items():
        print(f"  {k:<12} {v:7.1f} мс   {v / base:.2f}x к torch-only")


if __name__ == "__main__":
    main()
