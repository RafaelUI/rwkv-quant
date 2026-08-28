# -*- coding: utf-8 -*-
"""НЕДОСТАЮЩАЯ КЛЕТКА: torch fp32 С ВЕСАМИ ИЗ ФАЙЛА.

Клеток четыре, измерены три:

    torch fp32 + веса fake   0.002087   (fake_fp32_arm.py)
    MLX  fp32 + веса файла   0.003303   (kl_arms_realpath.py asis)
    MLX  fp32 без кванта     0.000002   (закон 38)
    torch fp32 + веса ФАЙЛА  -- ЗДЕСЬ

Сверка входов по ВСЕМ 798 ключам (`probe_fake_vs_file_all.py`) показала:
веса fake и файла бит-в-бит везде, кроме `g_lora`, и это расхождение
равно 0.0252% энергии ошибки квантования. Значит клетка обязана сесть на
0.002087 -- а если сядет на 0.0033, то весь остаток 0.001216 принадлежит
`g_lora`, вопреки его доле в энергии.

Разность этого плеча и `fake_fp32_arm` ЕСТЬ измеренная цена `g_lora`:
других различий в весах не осталось. То есть один прогон закрывает и
«послойную сверку», и «плечо с подменой g_lora».

Подменяется РОВНО weight у точек `comp.quant_points` (закон 27), файл и
чекпоинт те же, что во всех прочих плечах.

    RWKVQ_KL_REF=/tmp/ref32_1p5b_ml_8x512.npy \\
    python tests/_sess/file_weights_arm.py <файл.rwkvq>
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
from rwkv_quant.formats import reader  # noqa: E402
from rwkv_quant.models.rwkv7_ref import RWKV7Ref  # noqa: E402

PATH = sys.argv[1]
KEEP_FAKE = os.environ.get("RWKVQ_ARM_FAKE_GROUPS", "")


def load_file_weights(model, ck, cfg):
    """Веса из контейнера в модель. Возвращает статистику подмены."""
    from rwkv_quant.calibration import fake_quant
    keep = set(g for g in KEEP_FAKE.split(",") if g)
    done, skipped, moved = {}, [], 0.0
    nT = [0]
    for obj, attr, group, key in comp.quant_points(model):
        w = getattr(obj, attr)
        if w is None:
            continue
        qt = ck.tensors.get(key)
        if qt is None or qt.bits >= 16:
            skipped.append(key)
            continue
        if group in keep:
            out = fake_quant.q(w.to(torch.bfloat16) if os.environ.get("RWKVQ_ARM_FAKE_BF16") == "1" else w, group, cfg, key)
        else:
            out = reader.dequantize_banded(qt, torch.float32)
            # RWKV7Ref держит LoRA транспонированной относительно файла
            if tuple(out.shape) == tuple(w.shape)[::-1] != tuple(w.shape):
                out = out.T.contiguous()
                nT[0] += 1
        if tuple(out.shape) != tuple(w.shape):
            raise SystemExit("ФОРМА НЕ СОШЛАСЬ у %s: %s против %s"
                             % (key, tuple(out.shape), tuple(w.shape)))
        moved += float((out.to(torch.float32) - w.to(torch.float32))
                       .pow(2).sum())
        setattr(obj, attr, out.to(w.dtype))
        done[group] = done.get(group, 0) + 1
        del w, out
    return done, skipped, moved, nT[0]


def main():
    if not os.path.exists(A.REF):
        raise SystemExit("нет эталона %s" % A.REF)
    print("плечо fake=%s, файл %s, эталон %s"
          % (KEEP_FAKE or "нет", os.path.basename(os.path.realpath(PATH)),
             os.path.basename(A.REF)), flush=True)

    data, langs = A.load_data()
    data = data.to("cpu") if hasattr(data, "to") else data
    ref = np.load(A.REF, mmap_mode="r")
    model = RWKV7Ref(A.CKPT, device="cpu", dtype=torch.float32)
    ck = reader.load_raw(PATH)

    t0 = time.time()
    done, skipped, moved, nT = load_file_weights(model, ck, comp.CONFIGS["preset"]())
    print("  подменено %d тензоров за %.0f с (транспонировано %d), группы %s"
          % (sum(done.values()), time.time() - t0, nT, done), flush=True)
    print("  не квантованы в файле (%d): %s"
          % (len(skipped), ", ".join(skipped[:4])), flush=True)
    if not done:
        raise SystemExit("КОНТРОЛЬ: не подменено НИ ОДНОГО тензора")
    if moved == 0.0:
        raise SystemExit("КОНТРОЛЬ: подмена НИЧЕГО НЕ ИЗМЕНИЛА -- либо веса "
                         "уже были из файла, либо setattr не сработал")
    print("  контроль подмены: энергия смещения весов %.6e" % moved, flush=True)

    per_seq, tops = [], []
    for i, got in enumerate(A.logits_of(model, data)):
        k, t = A.kl_stats(np.asarray(ref[i]), got)
        per_seq.append(float(k.mean()))
        tops.append(float(t.mean()))
        print("  %d/%d KL=%.6f" % (i + 1, len(data), per_seq[-1]), flush=True)
    lo, hi = A.boot_ci(per_seq)
    print("KL = %.6f нат/токен  95%% CI [%.6f; %.6f]   top-1 %.3f%%"
          % (float(np.mean(per_seq)), lo, hi, 100 * np.mean(tops)))
    print("per_seq = %s" % [round(x, 6) for x in per_seq])


if __name__ == "__main__":
    main()
