"""ОДНОПРОХОДНЫЙ КЕРНЕЛЬ ДЕКВАНТА sb6 (K3): qblk -> fp16 [OUT, IN].

Зачем. На префилле плотный путь материализует веса в fp16, и штатная
цепочка MLX платит за это 146.0 мс при поле 30 (замеры 15-16.09):
concatenate удвоения кодов разрывает слияние и стоит 68.9 мс, а
оставшийся слитый кернель идёт на 48.8 ГБ/с против машинных 100+.
Здесь один проход: поток читает свой блок (16-24 Б), распаковывает в
регистрах и пишет 32 половины подряд. Цель числом: 146 -> 30-40 мс.

БИТ-В-БИТ -- ТРЕБОВАНИЕ, А НЕ ПОЖЕЛАНИЕ. Поэтому:
  1. распаковка НЕ переписана, а взята из штатного k3-кернеля
     (_K3_DECODE, _k3_plane) -- те же мультитрюки битплоскостей;
  2. масштаб и минимум считаются В MLX теми же строками, что в
     _dequant_w, и передаются готовыми: на них 1/16 и 1/128 объёма,
     оптимизировать нечего, а риск расхождения есть (закон 39);
  3. на lane остаётся ровно q * s + m в half.

ЧЕГО НЕ ЗНАЕМ ЗАРАНЕЕ: свернёт ли компилятор q*s+m в fma с ОДНИМ
округлением, тогда как MLX в своём слитом кернеле мог округлить дважды
(или наоборот). Это решает ГЕЙТ, а не рассуждение:
tests/test_gw_dequant_kernel_parity.py. Если бит-в-бит не выйдет -- это
расхождение в 1 ULP и решение владельца, а не повод подгонять порог.

MUTATE=1 портит кернель нарочно (сдвиг битплоскости 4 -> 5): гейт обязан
покраснеть, иначе он ничего не проверяет (закон 37).

ПЕРЕВОД СТРОКИ ТОЛЬКО ЧЕРЕЗ NL: экранирование в этом файле запрещено --
16.09 два патча молча положили в источник литеральный слэш-n, и Metal
падал на utils.h, а не на нашей строке (закон 33 про проверку записи).
"""
import os

import mlx.core as mx

from .quant_linear_gw import _K3_DECODE, _k3_plane

NL = chr(10)
_cache = {}
_REGS = ("l0", "l1", "l2", "l3", "h0", "h1", "h2", "h3")


def _src(xbits, mutate):
    parts = ["    uint gid = thread_position_in_grid.x;",
             "    if (gid >= OUT_C * NB) return;",
             "    uint row = gid / NB;",
             "    uint p   = gid % NB;",
             "    device const uint* qb = ((device const uint*)qblk)",
             "                          + (row * NB + p) * SU;",
             _K3_DECODE]
    if xbits >= 1:
        parts.append("    uint hb = qb[4];")
        parts.append(_k3_plane("hb", 5 if mutate else 4))
    if xbits >= 2:
        parts.append("    uint hb2 = qb[5];")
        parts.append(_k3_plane("hb2", 5))
    parts.append("    half s  = sc[row * NB + p];")
    parts.append("    half mn = mi[row * NB + p];")
    parts.append("    device half4* wp = (device half4*)(w + (row * NB + p) * 32);")
    parts.append("    float sf = (float)s;")
    for i, r in enumerate(_REGS):
        e = []
        for c in ("x", "y", "z", "w"):
            e.append("(half)((float)%s.%s * sf) + mn" % (r, c))
        parts.append("    wp[%d] = half4(%s);" % (i, ", ".join(e)))
    return NL.join(parts) + NL


def _kernel(IN, OUT, xbits, mutate):
    key = (IN, OUT, xbits, mutate)
    k = _cache.get(key)
    if k is None:
        hdr = NL.join(["",
                       "constant uint IN_C = %d;" % IN,
                       "constant uint OUT_C = %d;" % OUT,
                       "constant uint NB = %d;" % (IN // 32),
                       "constant uint SU = %d;" % (4 + xbits),
                       ""])
        k = mx.fast.metal_kernel(
            name="gw_dequant%d_%d_%d%s" % (4 + xbits, IN, OUT,
                                           "_mut" if mutate else ""),
            input_names=["qblk", "sc", "mi"],
            output_names=["w"],
            header=hdr, source=_src(xbits, mutate),
        )
        _cache[key] = k
    return k


def dequant_w(self):
    """Возвращает fp16 [OUT, IN]. Требует интерлив K3."""
    OUT, IN, NSB = self.out_features, self.in_features, self.NSB
    assert getattr(self, "_k3", False), "кернель декванта написан под интерлив K3"
    assert IN % 256 == 0, "IN кратен суперблоку 256"
    s = (self.qs.astype(mx.float32).reshape(OUT, NSB, 8)
         * self.d.astype(mx.float32)[..., None]).astype(mx.float16)
    m = (self.qm.astype(mx.float32).reshape(OUT, NSB, 8)
         * self.dm.astype(mx.float32)[..., None]).astype(mx.float16)
    kern = _kernel(IN, OUT, self.xbits, bool(int(os.environ.get("MUTATE", "0"))))
    return kern(
        inputs=[self.qblk, s.reshape(OUT, -1), m.reshape(OUT, -1)],
        grid=(OUT * (IN // 32), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(OUT, IN)], output_dtypes=[mx.float16],
    )[0]
