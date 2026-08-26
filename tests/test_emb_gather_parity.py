"""ГЕЙТ: emb-gather РАВЕН плотной таблице. Требуется РАВЕНСТВО, не порог.

Право требовать бит-в-бит здесь есть: gather считает ту же формулу теми же
округлениями (fp32 -> bf16 -> fp16), меняется только то, СКОЛЬКО строк
посчитано. Порядок суммирования не участвует -- в sym нет ни редукций, ни
накопления, только поэлементное умножение на масштаб блока.

Проверяется на РЕАЛЬНОМ файле (закон 17: синтетический случай, который
нельзя предъявить в настоящем чекпоинте, -- не покрытие, а фантазия) и на
индексах, которые ломают наивную реализацию:
  * дубли в одном батче (два одинаковых токена),
  * края словаря (0 и V-1),
  * форма [B, T] и форма [T] -- декод зовёт с одной, префилл с другой,
  * T=1 (декод) и T=512 (префилл).

    python tests/test_emb_gather_parity.py [model.rwkvq]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.backends.metal import quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"


def main():
    ck = load_raw(PATH)
    qt = ck.tensors["emb.weight"]
    V, D = qt.shape
    print("emb %s bits=%s mode=%s gs=%s sb=%s"
          % ((V, D), qt.bits, qt.gw_mode, qt.gw_gs, qt.gw_sb))

    dense = qm._dense(qt)
    qm.EMB_GATHER = True
    gath = qm._emb_table(qt)
    assert isinstance(gath, qm.SymGatherEmb), (
        "gather не применился -- гейт мерил бы плотный путь сам с собой")
    assert dense.dtype == gath.dtype == mx.float16
    print("плотная %s %s, gather %s %s"
          % (dense.shape, dense.dtype, gath.shape, gath.dtype))

    rs = np.random.RandomState(11)
    cases = {
        "декод T=1": np.array([[7]], dtype=np.int32),
        "префилл T=512": rs.randint(0, V, size=(1, 512)).astype(np.int32),
        "дубли": np.array([[5, 5, 5, 12345, 12345]], dtype=np.int32),
        "края словаря": np.array([[0, V - 1, 1, V - 2]], dtype=np.int32),
        "одномерный": rs.randint(0, V, size=(9,)).astype(np.int32),
        "батч B=2": rs.randint(0, V, size=(2, 40)).astype(np.int32),
    }

    ok = True
    for name, idx_np in cases.items():
        idx = mx.array(idx_np)
        a = np.array(dense[idx])
        b = np.array(gath[idx])
        same_shape = a.shape == b.shape
        eq = same_shape and bool((a == b).all())
        ok &= eq
        print("  %-15s форма %-18s %s"
              % (name, str(b.shape),
                 "равно" if eq else "РАСХОЖДЕНИЕ max|d|=%.3e" %
                 (np.abs(a.astype(np.float32) - b.astype(np.float32)).max()
                  if same_shape else float("nan"))))

    # НЕВЫРОЖДЕННОСТЬ (закон 8): равенство обязано быть содержательным.
    # Если бы обе ветки отдавали нули, всё выше было бы зелёным.
    idx = mx.array(np.array([[0, V - 1, 777]], dtype=np.int32))
    val = np.array(gath[idx]).astype(np.float32)
    assert np.abs(val).max() > 0, "gather отдал нули -- сравнивать нечего"
    nz = (np.abs(val) > 0).mean()
    print("  контроль: max|w| %.4f, ненулевых %.1f%%" % (np.abs(val).max(),
                                                         100 * nz))

    # ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ: подмена одного байта кода обязана быть видна.
    # Без него "равно" могло бы означать, что сравниваются две ссылки на
    # один и тот же буфер.
    c = np.array(gath.codes)
    c[0, 0] = np.int8(c[0, 0] + 7)
    gath.codes = mx.array(c)
    d0 = np.array(dense[mx.array(np.array([0], dtype=np.int32))])
    g0 = np.array(gath[mx.array(np.array([0], dtype=np.int32))])
    caught = not bool((d0 == g0).all())
    print("  отрицательный контроль (один байт кода изменён): %s"
          % ("расхождение видно" if caught else "НЕ ВИДНО -- гейт слеп"))
    ok &= caught

    print("\n[%s] emb-gather против плотной таблицы" % ("OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
