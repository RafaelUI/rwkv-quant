# -*- coding: utf-8 -*-
"""FAKE-ПУТЬ В fp32: ПОСЛЕДНЕЕ ПЛЕЧО ПРИОРИТЕТА 2.

Установлено 27.08: веса fake-пути и веса файла БИТ-В-БИТ совпадают во всех
группах, кроме `g_lora` (rtn@8, расхождение 13.3% энергии ошибки кванта), а
множества квантованных ключей различаются на два неиспользуемых тензора
(`blocks.0.att.v1/v2`). Значит остаток 0.000541 нат/ток не может быть ни
контейнером, ни кернелем, ни весами -- остаётся ОДНО различие: fake считает
`RWKV7Ref` на MPS в bf16, реальный путь -- MLX в fp32.

Здесь fake считается ТЕМ ЖЕ кодом, но в fp32 на CPU -- ровно как эталон.
Если KL сядет на реальный путь (0.0033), остаток разобран целиком и он
принадлежит ЛИНЕЙКЕ, а не измеряемому (закон 38 второй раз, теперь на
измеряемом плече).

    RWKVQ_KL_REF=/tmp/ref32_1p5b_ml_8x512.npy python tests/_sess/fake_fp32_arm.py [конфиг]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ablate_subgroups as A  # noqa: E402
import ablate_sym_composite as comp  # noqa: E402
from rwkv_quant.models.rwkv7_ref import RWKV7Ref  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "preset"
DT = os.environ.get("RWKVQ_ARM_DTYPE", "fp32")


def main():
    if not os.path.exists(A.REF):
        raise SystemExit("нет эталона %s" % A.REF)
    dev, dt = (("cpu", torch.float32) if DT == "fp32"
               else ("mps", torch.bfloat16))
    print("плечо fake/%s на %s, эталон %s"
          % (DT, dev, os.path.basename(A.REF)), flush=True)

    data, langs = A.load_data()
    data = data.to(dev) if hasattr(data, "to") else data
    ref = np.load(A.REF, mmap_mode="r")
    model = RWKV7Ref(A.CKPT, device=dev, dtype=dt)

    t0 = time.time()
    done = A.prequantize(model, comp.CONFIGS[NAME](), lambda k: True)
    if not done:
        raise SystemExit("КОНТРОЛЬ: не квантовано НИ ОДНОГО тензора")
    print("  квантовано по группам %s за %.0f с" % (done, time.time() - t0),
          flush=True)

    per_seq, tops = [], []
    for i, got in enumerate(A.logits_of(model, data)):
        k, t = A.kl_stats(np.asarray(ref[i]), got)
        per_seq.append(float(k.mean())); tops.append(float(t.mean()))
        print("  %d/%d KL=%.6f" % (i + 1, len(data), per_seq[-1]), flush=True)
    lo, hi = A.boot_ci(per_seq)
    print("KL = %.6f нат/токен  95%% CI [%.6f; %.6f]   top-1 %.3f%%"
          % (float(np.mean(per_seq)), lo, hi, 100 * np.mean(tops)))
    print("per_seq = %s" % [round(x, 6) for x in per_seq])


if __name__ == "__main__":
    main()
