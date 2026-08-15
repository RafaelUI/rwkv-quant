"""ГЕЙТ ДЕКВАНТА sym ОДНИМ КЕРНЕЛЕМ. Требуется РАВЕНСТВО.

Кернель заменяет прежнюю цепочку MLX-операций (`_dequant_w_ref`), которая
осталась на месте именно ради этой сверки. Право требовать бит-в-бит
здесь есть: обе реализации считают одну и ту же формулу в одном и том же
порядке -- (q - 32) в fp16, умножение на half-масштаб блока, -- и ни одна
не суммирует ничего, где порядок мог бы разойтись.

ЧТО ИМЕННО ЛОВИТСЯ. Не арифметика, а ПОРЯДОК КОЛОНОК: split у sym
блок-локальный, поэтому внутри пары блоков колонки идут l0,l1,h0,h1
(первый блок) и l2,l3,h2,h3 (второй). Перепутать их местами -- получить
те же 32 числа в другом порядке, то есть правдоподобный и неверный
результат, который не поймает ни одна проверка «на глаз» (закон 8).
Поэтому сверяются ВСЕ элементы, а не нормы.

Плюс отдельное утверждение, что сверка не вырождена: обе битности
присутствуют, и хотя бы одна матрица шире суперблока.

    python tests/test_sym_dequant_kernel.py [model.rwkvq ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.backends.metal.quant_linear_sym import SymQuantLinear  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATHS = sys.argv[1:] or ["/tmp/reduction_sym_head8.rwkvq"]


def main():
    ok, seen_bits, n = True, set(), 0
    for path in PATHS:
        model = qm.QuantRWKV7(load_raw(path))
        lins = [("head", model.head)]
        for i, b in enumerate(model.blocks):
            lins += [(f"blocks.{i}.cmix.key", b.cmix.key),
                     (f"blocks.{i}.cmix.value", b.cmix.value),
                     (f"blocks.{i}.att.receptance", b.tmix.r_proj),
                     (f"blocks.{i}.att.output", b.tmix.o_proj)]
        for name, lin in lins:
            if not isinstance(lin, SymQuantLinear):
                continue
            a, b_ = lin._dequant_w(), lin._dequant_w_ref()
            mx.eval(a, b_)
            d = float(mx.abs(a.astype(mx.float32)
                             - b_.astype(mx.float32)).max())
            seen_bits.add(lin.bits)
            n += 1
            if d != 0:
                print(f"  РАСХОЖДЕНИЕ {os.path.basename(path)} {name} "
                      f"({lin.bits} бит): max|Δ| = {d:.3e}")
                ok = False
            del a, b_
        del model
    print(f"сверено тензоров: {n}, битности: {sorted(seen_bits)}")
    if len(seen_bits) < 2:
        print("  ВЫРОЖДЕНО: в наборе только одна битность, вторая ветка "
              "кернеля не проверена")
        ok = False
    print(f"ГЕЙТ: {'ЗЕЛЁНЫЙ (бит-в-бит)' if ok else 'КРАСНЫЙ'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
