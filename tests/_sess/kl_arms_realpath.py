# -*- coding: utf-8 -*-
"""ГДЕ ЖИВУТ ОСТАВШИЕСЯ 0.000541 нат/ток РЕАЛЬНОГО ПУТИ ПОВЕРХ fake (приоритет 2).

С честным fp32-эталоном (закон 38) на 1.5B: fake-путь 0.002762, реальный
путь 0.003303. Разность 0.000541 -- единственный неразобранный остаток
бюджета REDUCTION. Записанные кандидаты: fp16-контейнеры плотных тензоров,
арифметика GEMV-кернеля, fp16-голова.

ПЛЕЧИ СТРОЯТСЯ ПОДМЕНОЙ ОДНОЙ ФУНКЦИИ (закон 27), не версией файла и не
пересборкой .rwkvq: файл во всех плечах ОДИН И ТОТ ЖЕ, меняется только то,
чем его читают.

    asis      -- как есть, умолчания (gather включён)
    mm32      -- `_mm` считает в fp32. Плотные матмулы (LoRA-ветки, g_lora)
                 перестают приводить АКТИВАЦИИ к fp16; веса те же fp16.
                 Изолирует АРИФМЕТИКУ, не хранение.
    dense32   -- `_dense` отдаёт fp32 (+ gather выключен, чтобы emb тоже
                 пошёл через `_dense`). Изолирует ХРАНЕНИЕ в fp16 вместе с
                 его арифметикой: dense32 - mm32 = цена самого контейнера.
    nokernel  -- все квантованные Linear заменены плотным fp32-деквантом
                 читалки. Кернеля нет вовсе, веса становятся РОВНО теми же,
                 что у fake-пути (деквант читалки, округление в bf16), а
                 активации остаются fp32. ОЖИДАНИЕ: KL сядет на fake
                 (0.002762). Если сядет -- остаток разобран целиком, и это
                 же служит проверкой самой лестницы.

Одно плечо на процесс (законы 2 и 13). Пары по последовательностям и
парный бутстрэп -- в `kl_real_path.py --report` (закон 35).

    RWKVQ_KL_REF=/tmp/ref32_1p5b_ml_8x512.npy \\
    python tests/_sess/kl_arms_realpath.py <файл.rwkvq> <asis|mm32|dense32|nokernel>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

PATH = sys.argv[1]
ARM = sys.argv[2]

# ЧИТАЕТСЯ ПРИ ИМПОРТЕ МОДУЛЯ, поэтому ставится ДО него.
if ARM in ("dense32", "nokernel"):
    os.environ["RWKVQ_EMB_GATHER"] = "0"

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

import rwkv_quant.backends.metal.quant_model as QM  # noqa: E402
import rwkv_quant.formats.reader as R  # noqa: E402

HITS = {"_mm": 0, "_dense": 0, "_linear": 0}
DTYPES = {"_dense": set(), "_linear": set()}


def patch_mm32():
    def _mm(x, w):
        HITS["_mm"] += 1
        return (x.astype(mx.float32) @ w.astype(mx.float32).T).astype(x.dtype)
    QM._mm = _mm


def patch_dense32():
    def _dense(qt):
        HITS["_dense"] += 1
        if qt.bits >= 16:
            a = mx.array(qt.dense.to(torch.float32).numpy())
        else:
            a = mx.array(R.dequantize_banded(qt, torch.float32).numpy())
        DTYPES["_dense"].add(str(a.dtype))
        return a
    QM._dense = _dense


def patch_nokernel():
    def _linear(qt):
        HITS["_linear"] += 1
        if qt.bits < 16:
            w = R.dequantize_banded(qt, torch.float32)
        else:
            w = qt.dense.to(torch.float32)
        a = mx.array(w.numpy())
        DTYPES["_linear"].add(str(a.dtype))
        return QM._DenseLinear(a)
    QM._linear = _linear


def control_mm():
    """КОНТРОЛЬ ВКЛЮЧЕНИЯ ДЛЯ mm32, ФУНКЦИОНАЛЬНЫЙ. «Патч не применился» и
    «эффекта нет» по выходу модели неразличимы, поэтому подмена проверяется
    прямо: вход, у которого fp16 отъедает разряд, обязан дать РАЗНЫЕ числа
    у прежней и новой реализации."""
    x = mx.array([[1.0 + 2.0 ** -13]], dtype=mx.float32)
    w = mx.ones((1, 1), dtype=mx.float16)
    old = float((x.astype(w.dtype) @ w.T).astype(x.dtype).item())
    new = float(QM._mm(x, w).item())
    print("  контроль mm: прежний %.9f, патченый %.9f" % (old, new))
    if old == new:
        raise SystemExit("КОНТРОЛЬ ВКЛЮЧЕНИЯ НЕ ПРОШЁЛ: _mm не подменён")


def control_tensor(fn, key="emb.weight"):
    """Контроль для dense32/nokernel: патч зовут на НАСТОЯЩЕМ тензоре файла
    и требуют fp32 на выходе. Проверяется до прогона, а не после."""
    ck = R.load_raw(PATH)
    qt = ck.tensors[key]
    a = fn(qt)
    a = getattr(a, "w", a)
    print("  контроль тензора %s: dtype %s" % (key, a.dtype))
    if a.dtype != mx.float32:
        raise SystemExit("КОНТРОЛЬ ВКЛЮЧЕНИЯ НЕ ПРОШЁЛ: dtype %s" % a.dtype)
    del ck, qt, a


if ARM == "asis":
    pass
elif ARM == "mm32":
    patch_mm32()
    control_mm()
elif ARM == "dense32":
    patch_dense32()
    control_tensor(QM._dense)
elif ARM == "nokernel":
    patch_dense32()
    patch_nokernel()
    control_tensor(QM._dense)
    control_tensor(QM._linear, "head.weight")
else:
    raise SystemExit("неизвестное плечо %r" % ARM)

HITS.update({k: 0 for k in HITS})

import kl_real_path as K  # noqa: E402

K.run(PATH, ARM, False)
print("  вызовов: _mm=%d _dense=%d _linear=%d, dtypes %s"
      % (HITS["_mm"], HITS["_dense"], HITS["_linear"],
         {k: sorted(v) for k, v in DTYPES.items() if v}))
if ARM == "mm32" and HITS["_mm"] == 0:
    raise SystemExit("патч _mm ни разу не позван -- плечо недействительно")
if ARM in ("dense32", "nokernel") and HITS["_dense"] == 0:
    raise SystemExit("патч _dense ни разу не позван -- плечо недействительно")
if ARM == "nokernel" and HITS["_linear"] == 0:
    raise SystemExit("патч _linear ни разу не позван -- плечо недействительно")
