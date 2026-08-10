"""Цена схем в БИТАХ НА ВЕС -- измеренная, не выведенная.

Поиск калибровки минимизирует размер, поэтому его модель стоимости
обязана совпадать с тем, что writer реально кладёт на диск. Здесь
считается сумма байт всех буферов QuantizedTensor -- ровно те поля, что
уезжают в safetensors (writer.TENSOR_FIELDS).

Гейт-режим (--check) сверяет измеренное с константами в
calibration/schema_space.py и краснеет при расхождении: если кто-то
добавит упаковку в asym-ветку или нибблы для rtn@6, поиск обязан узнать
об этом сразу, а не продолжать оптимизировать вымышленный размер.

    python tests/probe_schema_cost.py
    python tests/probe_schema_cost.py --check
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration import schema_space as ss
from rwkv_quant.formats import writer
from rwkv_quant.formats.writer import TENSOR_FIELDS

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
KEY = "blocks.3.ffn.key.weight"


def cost(key, w, cfg):
    qt = writer.quantize_tensor(key, w, cfg, real_gw=True)
    total, fields = 0, []
    for f in TENSOR_FIELDS:
        v = getattr(qt, f, None)
        if v is None or v.numel() == 0:
            continue
        total += v.numel() * v.element_size()
        fields.append(f"{f}:{str(v.dtype).replace('torch.','')}")
    return total * 8 / w.numel(), total, fields


def main(check=False):
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    w = sd[KEY].to(torch.bfloat16).contiguous()
    IN = int(w.shape[1])
    print(f"{KEY} {tuple(w.shape)} = {w.numel()} весов\n")
    print(f"{'схема':22s} {'бит/вес':>9s} {'модель':>9s}  поля")
    bad = []

    def row(tag, cfg, model):
        eff, _, fields = cost(KEY, w, cfg)
        ok = abs(eff - model) < 1e-3
        if not ok:
            bad.append((tag, eff, model))
        print(f"{tag:22s} {eff:9.3f} {model:9.3f}{'' if ok else '  <-- РАСХОЖДЕНИЕ'}"
              f"  {' '.join(fields)}")

    for bits in (4, 5, 6):
        row(f"sb6@{bits}",
            QuantConfig(cmix=bits, group_scale={"cmix": 32},
                        group_scale_mode={"cmix": "asym_sb6"}),
            ss.SB6_COST[bits])
    for bits in (5, 6, 8):
        row(f"asym gw64@{bits}",
            QuantConfig(cmix=bits, group_scale={"cmix": 64},
                        group_scale_mode={"cmix": "asym"}),
            ss.ASYM_COST)
    for bits in (4, 6, 8):
        row(f"rtn per-row@{bits}", QuantConfig(cmix=bits),
            ss._rtn_cost(bits, IN))

    print("\nПорядок кандидатов, который из этого следует (IN=%d):" % IN)
    for c in ss.candidates_for(IN, have_act_stats=False):
        print(f"   {c}")

    if check:
        assert not bad, ("модель стоимости разошлась с writer:\n  " +
                         "\n  ".join(f"{t}: измерено {e:.3f}, в модели {m:.3f}"
                                     for t, e, m in bad))
        print("\nМОДЕЛЬ СТОИМОСТИ СОВПАДАЕТ С WRITER")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
