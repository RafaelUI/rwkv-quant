"""ГЕЙТ ФЬЮЗА r/k/v ДЛЯ РАСКЛАДКИ sym.

Два утверждения, оба требуют РАВЕНСТВА, и второе объясняет, зачем нужно
первое.

  1. НЕФЬЮЗНУТЫЙ кернель не изменился. Фьюз сделан не отдельной копией
     кернеля, а параметром OUT_PER в ТОМ ЖЕ исходнике: при OUT_PER=OUT_C
     индекс входа вырождается в прежний. Это правильный выбор (закон 23:
     параллельные реализации расходятся ровно в тот день, когда правку
     вносят в одну), но он означает, что источник ПЕРЕГЕНЕРИРОВАН, и
     «там же ничего не поменялось» надо доказать, а не заявить. Поэтому
     выходы GEMV заморожены В ФАЙЛ до правки и сверяются после.
  2. ФЬЮЗНУТЫЙ вызов равен K отдельным. Фьюз не меняет ни порядок
     суммирования, ни данные -- он конкатенирует строки и выбирает вход
     по номеру строки, -- поэтому здесь законно требовать бит-в-бит, а
     не порог. Если однажды станет неравно, значит фьюз считает не то же
     самое, и «ускорение без потери качества» перестало быть таковым.

    python tests/test_sym_fuse_parity.py --freeze <model.rwkvq>   # ДО правки
    python tests/test_sym_fuse_parity.py <model.rwkvq>            # после

Эталон: /tmp/sym_kernel_ref.npz (мелкий, только выходы GEMV).
"""
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.backends.metal.quant_linear_sym import SymQuantLinear  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

REF = os.environ.get("RWKVQ_SYM_REF", "/tmp/sym_kernel_ref.npz")

# формы берутся из НАСТОЯЩЕГО чекпоинта, а не выдумываются (закон 17)
KEYS = [
    "blocks.0.att.receptance.weight",
    "blocks.0.att.key.weight",
    "blocks.0.att.value.weight",
    "blocks.0.att.output.weight",
    "blocks.3.ffn.key.weight",
    "blocks.3.ffn.value.weight",
    "head.weight",
]
NS = (1, 3, 16)


def sym_tensors(ckpt):
    return {k: qt for k, qt in ckpt.tensors.items()
            if getattr(qt, "gw_mode", "") == "sym"}


def outputs(lin, key):
    """Выходы GEMV на фиксированном входе. Сид зависит от ключа, чтобы
    разные тензоры не сверялись одним и тем же вектором.

    СИД БЕРЁТСЯ ИЗ crc32, А НЕ ИЗ hash(). `hash()` для строк в питоне
    рандомизирован ПО ПРОЦЕССУ (PYTHONHASHSEED), поэтому первая версия
    гейта заморозила эталон на одном входе, а сверяла на другом -- и
    показала расхождение во ВСЕХ 21 выходе с max|Δ| порядка самой
    величины. Выглядело это ровно как сломанный кернель. Мораль в общем
    виде: если эталон переживает процесс, то и всё, что его определяет,
    обязано переживать процесс."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    res = {}
    for n in NS:
        x = mx.array(rng.standard_normal((n, lin.in_features)).astype(np.float32))
        y = lin._gemv(x, n)
        mx.eval(y)
        res[f"{key}|{n}"] = np.array(y)
    return res


def freeze(path):
    ckpt = load_raw(path)
    tens = sym_tensors(ckpt)
    got, used = {}, []
    for k in KEYS:
        if k not in tens:
            continue
        lin = SymQuantLinear(tens[k])
        got.update(outputs(lin, k))
        used.append(f"{k}@{lin.bits} {tens[k].shape}")
        del lin
    if not got:
        print("ПРОВАЛ: в чекпоинте нет ни одного sym-тензора из списка")
        sys.exit(1)
    np.savez(REF, **got)
    print(f"эталон записан: {REF}, {len(got)} выходов")
    for u in used:
        print(f"  {u}")


def check(path):
    if not os.path.exists(REF):
        print(f"ПРОВАЛ: нет эталона {REF}. Сначала --freeze ДО правки кернеля")
        sys.exit(1)
    ref = np.load(REF)
    ckpt = load_raw(path)
    tens = sym_tensors(ckpt)
    fails, n_ok = [], 0

    # --- 1. нефьюзнутый кернель против замороженного эталона
    for k in KEYS:
        if k not in tens:
            continue
        lin = SymQuantLinear(tens[k])
        for name, val in outputs(lin, k).items():
            if name not in ref:
                fails.append(f"{name}: нет в эталоне")
                continue
            if not np.array_equal(val, ref[name]):
                d = np.abs(val - ref[name])
                fails.append(f"{name}: кернель разошёлся с эталоном, "
                             f"max|Δ|={d.max():.3e}, "
                             f"{int((val != ref[name]).sum())} элементов")
            else:
                n_ok += 1
        del lin
    print(f"нефьюзнутый против эталона: {n_ok} выходов сверено")

    # --- 2. фьюз против K отдельных вызовов
    from rwkv_quant.backends.metal.quant_linear_sym import SymQuantLinearFused
    groups = []
    for pre in ("blocks.0.att.", "blocks.5.att.", "blocks.11.att."):
        keys = [pre + n for n in ("receptance.weight", "key.weight",
                                  "value.weight")]
        if all(k in tens for k in keys):
            groups.append(keys)
    if not groups:
        print("ПРОВАЛ: не нашлось ни одной тройки r/k/v в sym -- "
              "утверждение 2 вырождено (закон 17)")
        sys.exit(1)

    n_f = 0
    for keys in groups:
        lins = [SymQuantLinear(tens[k]) for k in keys]
        fused = SymQuantLinearFused(lins)
        rng = np.random.default_rng(7)
        xs = mx.array(rng.standard_normal(
            (len(lins), lins[0].in_features)).astype(np.float32))
        got = np.array(fused(xs))
        want = np.stack([np.array(l._gemv(xs[i:i + 1], 1))[0]
                         for i, l in enumerate(lins)])
        if got.shape != want.shape:
            fails.append(f"{keys[0]}: форма {got.shape} против {want.shape}")
        elif not np.array_equal(got, want):
            d = np.abs(got - want)
            fails.append(f"{keys[0]}: фьюз разошёлся с отдельными вызовами, "
                         f"max|Δ|={d.max():.3e}, "
                         f"{int((got != want).sum())} элементов")
        else:
            n_f += 1
        del lins, fused
    print(f"фьюз против отдельных вызовов: {n_f} троек сверено")

    if fails:
        for f in fails[:20]:
            print(f"  ПРОВАЛ {f}")
        print(f"\nГЕЙТ КРАСНЫЙ: {len(fails)} расхождений")
        sys.exit(1)
    print("\nГЕЙТ ЗЕЛЁНЫЙ")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--freeze"]
    path = args[0] if args else "/tmp/reduction_sym_head8.rwkvq"
    if "--freeze" in sys.argv:
        freeze(path)
    else:
        check(path)
