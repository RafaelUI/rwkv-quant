"""Численная сверка SymQuantLinear (GEMV sym + GEMM-путь) с референсом
x @ dequant(qt).T на ЖИВЫХ тензорах чекпоинта, для шести и восьми бит.

Аналог test_gw_kernel_int6.py для симметричной раскладки. Проверяется
ровно то, что кернель считает ту же матрицу, что лежит в файле: эталон
берётся из reader._dequantize_one, то есть из той же функции, которой
меряется ppl. Порог 3e-3 -- как у sb6-гейтов; кернель суммирует в другом
порядке, чем плотный матмул, и требовать равенства здесь нельзя (в
отличие от гейта упаковщика, где равенство обязательно).

ЧТО ЭТОТ ГЕЙТ ЛОВИТ В ПЕРВУЮ ОЧЕРЕДЬ. Порядок регистров при шести битах:
split блок-локальный, блок 16, поэтому колонки идут l0,l1,h0,h1 /
l2,l3,h2,h3, а не как при gs=32. Перепутанный порядок даёт ПРАВДОПОДОБНЫЙ
результат -- те же числа, переставленные внутри блока, -- и ловится
только сверкой с эталоном.

    python tests/test_sym_kernel.py [ckpt]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal.quant_linear_sym import SymQuantLinear  # noqa: E402
from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.formats.reader import _dequantize_one  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
MAX_ROWS = 8192

CASES = [
    ("blocks.0.att.receptance.weight", "proj"),   # IN=2048  OUT=2048
    ("blocks.0.ffn.key.weight", "cmix"),          # IN=2048  OUT=8192
    ("blocks.0.ffn.value.weight", "cmix"),        # IN=8192  OUT=2048
    ("head.weight", "head"),                      # IN=2048  OUT=65536 (срез)
]
FAILS = []


def main():
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    torch.manual_seed(0)
    stats = None
    for cand in sorted(__import__("glob").glob("/tmp/act_stats_*.pt")):
        st = torch.load(cand)
        v = st.get("blocks.0.ffn.key.weight")
        if v is not None and int(v.numel()) == int(sd["emb.weight"].shape[1]):
            stats = cand
            break
    print(f"act_stats: {stats}")

    for key, group in CASES:
        if key not in sd:
            continue
        w = sd[key]
        if w.shape[0] > MAX_ROWS:
            w = w[:MAX_ROWS]
        w = w.contiguous()
        for bits in (8, 6):
            cfg = QuantConfig(**{group: bits}, group_scale={group: 16},
                              group_scale_mode={group: "sym_aw"},
                              act_stats_path=stats)
            qt = quantize_tensor(key, w, cfg, real_gw=True)
            assert qt.gw_mode == "sym" and qt.bits == bits
            lin = SymQuantLinear(qt)
            ref_w = _dequantize_one(qt).float().numpy()
            for N in (1, 3, 16, 128):
                x = torch.randn(N, qt.shape[1]).numpy().astype(np.float32)
                y = np.array(lin(mx.array(x)))
                ref = x @ ref_w.T
                rel = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-9)
                path = "GEMM" if N >= 128 else "GEMV"
                ok = rel < 3e-3
                if not ok:
                    FAILS.append(f"{key}@{bits} N={N} rel={rel:.3e}")
                print(f"  {'ok  ' if ok else 'FAIL'} {key:32s} @{bits} "
                      f"{tuple(qt.shape)} N={N:3d} {path} relmax={rel:.3e}")
            del qt, lin, ref_w
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else "ПРОВАЛЕН: " + "; ".join(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
