"""ГЕЙТ ЯДРА wkv7_infer: РАВЕНСТВО, а не порог.

Правка стейджинга не меняет ни арифметику, ни ПОРЯДОК СУММИРОВАНИЯ: те же
64 FMA в том же порядке, только операнды приезжают из threadgroup-памяти
вместо глобальной. Значит право требовать бит-в-бит здесь есть -- в
отличие от реассоциации (ACC), которую этот гейт как раз и не пропустит.

    python test_wkv_infer_parity.py --freeze     # ДО правки, на чистом дереве
    python test_wkv_infer_parity.py              # ПОСЛЕ правки

Эталон: /tmp/wkv_infer_ref.npz. Сид входов -- zlib.crc32, а не hash():
hash() для строк рандомизирован по процессу, и «до» с «после» считались бы
на РАЗНЫХ входах (закон 28).
"""
import os
import sys
import zlib

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_metal.kernel.wkv7 import wkv7_infer, HEAD_SIZE  # noqa: E402

# ЛОВУШКА (закон 14): в ~/Develop/tests/venv лежит НЕ-editable копия
# rwkv_metal от 25.07, отставшая на несколько правок. Скрипт, импортировавший
# rwkv_metal до rwkv_quant, получил бы её молча. Поэтому путь вставляется
# первым, а исполняемый файл ПЕЧАТАЕТСЯ.
import rwkv_metal.kernel as _km  # noqa: E402
print("исполняется rwkv_metal:", __import__("sys").modules["rwkv_metal"].__file__)

REF = "/tmp/wkv_infer_ref.npz"
FREEZE = "--freeze" in sys.argv

# (B, T, H): T=1 -- декод, 3/16 -- верификация и чанк, 512 -- префилл;
# H=32 -- 1.5B, H=40 -- 2.9B, H=16 -- 0.4B векторная; B=2 -- батч.
CASES = [(1, 1, 32), (1, 3, 32), (1, 16, 32), (1, 512, 32),
         (1, 1, 40), (1, 512, 40), (1, 512, 16), (2, 128, 32), (1, 7, 32)]


def inputs(B, T, H):
    D = HEAD_SIZE
    seed = zlib.crc32(f"wkv7-infer-{B}-{T}-{H}".encode()) & 0x7FFFFFFF
    rs = np.random.RandomState(seed)

    def g(scale=1.0):
        return mx.array((rs.randn(B, T, H, D) * scale).astype(np.float32))

    r = g(); k = g(); v = g(); a = g(0.1); b = g(0.1)
    w = mx.array((0.85 + 0.14 * rs.rand(B, T, H, D)).astype(np.float32))
    h = mx.array((rs.randn(B, H, D, D) * 0.05).astype(np.float32))
    return r, w, k, v, a, b, h


def run(case):
    out, h_out = wkv7_infer(*inputs(*case))
    return np.asarray(out), np.asarray(h_out)


if FREEZE:
    data = {}
    for c in CASES:
        o, h = run(c)
        key = "_".join(map(str, c))
        data[f"out_{key}"] = o
        data[f"h_{key}"] = h
    np.savez_compressed(REF, **data)
    print(f"заморожено {len(CASES)} случаев -> {REF} "
          f"({os.path.getsize(REF)/1e6:.1f} МБ)")
    sys.exit(0)

if not os.path.exists(REF):
    sys.exit(f"НЕТ ЭТАЛОНА {REF}: сначала --freeze на дереве ДО правки")

ref = np.load(REF)
bad = 0
print(f"{'случай (B,T,H)':<18}{'max|dY|':>12}{'max|dH|':>12}  вердикт")
for c in CASES:
    key = "_".join(map(str, c))
    o, h = run(c)
    d1 = float(np.abs(o - ref[f"out_{key}"]).max())
    d2 = float(np.abs(h - ref[f"h_{key}"]).max())
    ok = (d1 == 0.0 and d2 == 0.0)
    bad += not ok
    print(f"{str(c):<18}{d1:>12.3e}{d2:>12.3e}  {'РАВНО' if ok else 'РАСХОЖДЕНИЕ'}")

# Контроль вырожденности: эталон не должен быть нулями (иначе гейт зелён всегда)
amp = max(float(np.abs(ref[f"out_{'_'.join(map(str, c))}"]).max()) for c in CASES)
print(f"\nамплитуда эталона {amp:.3e} "
      f"({'ненулевой' if amp > 1e-3 else 'ВЫРОЖДЕН -- гейт ничего не проверяет'})")
print("ГЕЙТ ЗЕЛЁНЫЙ" if bad == 0 and amp > 1e-3 else f"ГЕЙТ КРАСНЫЙ ({bad} случаев)")
sys.exit(1 if bad else 0)
