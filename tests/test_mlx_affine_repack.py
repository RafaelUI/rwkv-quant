"""
Гейт перекладки sb6 в родной контейнер MLX: числа не меняются.

Это утверждение сильнее, чем «расхождение мало». Наша формула
w = q*s + m при беззнаковых кодах и блоке 32 совпадает с affine-моделью
MLX буква в букву, поэтому перекладка обязана быть ТОЖДЕСТВЕННОЙ: те же
коды, тот же scale, тот же bias, другой лишь порядок бит. Если бы
пришлось звать mx.quantize по деквантованному весу, калибровка бы
потерялась (пересчёт scale/bias по min/max блока), и весь смысл
пресетов вместе с ней.

Проверяется на РЕАЛЬНОМ чекпоинте, по трём осям:
  1. коды после round-trip через контейнер MLX -- целочисленно те же;
  2. mx.dequantize из нового контейнера против codec.dequant_sb6 --
     сверяются коды, восстановленные обратным ходом, а не значения:
     ядро округляет по-своему, и сравнение значений в fp16 даёт ложные
     75-81% на заведомо верной раскладке (см. probe_mlx_native_packing);
  3. значения -- в пределах одного ulp fp16, как контроль, что дело
     действительно в округлении, а не в подмене чисел.

Плюс режим --dump: кладёт эталон для порта в Swift.

    python tests/test_mlx_affine_repack.py <model.rwkvq> [--dump out.safetensors]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.formats import codec  # noqa: E402

PER_SHAPE = 2          # тензоров на форму: формы различаются, тензоры нет
DUMP_ROWS = 64         # строк на тензор в фикстуре
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def main():
    path = sys.argv[1]
    dump = None
    if "--dump" in sys.argv:
        dump = sys.argv[sys.argv.index("--dump") + 1]

    manifest, arrays = codec.open_rwkvq(path)
    sb6 = [(k, m) for k, m in manifest["tensors"].items() if m["kind"] == "sb6"]
    print(f"{os.path.basename(path)}: sb6-тензоров {len(sb6)}")

    # по нескольку представителей на форму: разные формы -- разные
    # раскладки грида, один тензор на форму этого не покрывает
    seen, picked = {}, []
    for k, m in sorted(sb6):
        sh = tuple(m["shape"])
        if len(seen.setdefault(sh, [])) < PER_SHAPE:
            seen[sh].append(k)
            picked.append((k, m))

    fixture, n_el = {}, 0
    n_dead_total, n_blocks_total = [0], [0]
    for key, m in picked:
        OUT, IN = m["shape"]
        gs, sb = m["gw_gs"], m["gw_sb"]

        def buf(f):
            return arrays.get(f"{key}::{f}")

        wq, scales, biases, bits = codec.sb6_to_mlx_affine(
            buf("codes_packed"), buf("gw_qsqm"), buf("gw_d"), buf("gw_dm"),
            shape=(OUT, IN), gs=gs, sb=sb, nb=m.get("n_blocks"),
            qh=buf("gw_qh"), qh2=buf("gw_qh2"))
        NB = IN // gs

        # 1. round-trip кодов внутри нашей же раскладки
        back = codec.unpack_mlx_affine(wq, bits, NB)
        q_ref = codec.unpack_nib_block(buf("codes_packed"), gs).astype(np.uint32)
        if buf("gw_qh") is not None:
            q_ref += codec.unpack_bitplane(buf("gw_qh"), IN).astype(np.uint32) * 16
        if buf("gw_qh2") is not None:
            q_ref += codec.unpack_bitplane(buf("gw_qh2"), IN).astype(np.uint32) * 32
        ok1 = bool((back.reshape(OUT, IN) == q_ref).all())

        # 2. что из контейнера прочитает САМО ядро
        got = np.array(mx.dequantize(mx.array(wq), mx.array(scales),
                                     mx.array(biases), group_size=32, bits=bits)
                       .astype(mx.float32)).reshape(OUT, NB, 32)
        s32 = scales.astype(np.float32)[..., None]
        b32 = biases.astype(np.float32)[..., None]
        # ВЫРОЖДЕННЫЕ БЛОКИ. Writer клипует scale снизу числом 1e-8, но
        # в fp16 оно не представимо (минимальная субнормаль ~6e-8) и
        # превращается в ноль. Значит у блока, где qs*d ушло под fp16,
        # контейнер MLX хранит scale = 0, а эталон считает с 1e-8.
        # Коды такого блока из значений не восстановить в принципе
        # (все веса равны bias), поэтому они исключаются из проверки
        # кодов и учитываются отдельно -- молча делить на ноль и
        # получать inf было бы проверкой ни о чём.
        live = (s32 != 0)
        n_dead = int((~live).sum())
        codes_from_mlx = np.rint(np.where(live, (got - b32) / np.where(live, s32, 1),
                                          q_ref.reshape(OUT, NB, 32))).astype(np.int64)
        ok2 = bool((codes_from_mlx == q_ref.reshape(OUT, NB, 32)).all())

        # 3. значения против нашего эталонного декванта
        ref = codec.dequant_sb6(
            buf("codes_packed"), buf("gw_qsqm"), buf("gw_d"), buf("gw_dm"),
            shape=(OUT, IN), gs=gs, sb=sb, nb=m.get("n_blocks"),
            qh=buf("gw_qh"), qh2=buf("gw_qh2"))
        g = got.reshape(OUT, IN)
        # допуск: ulp fp16 плюс вклад вырожденных блоков. У них эталон
        # считает q*1e-8, а контейнер -- q*0, то есть расхождение не
        # больше 63e-8 по модулю и никакого практического смысла не
        # имеет; но записать его надо, а не спрятать в относительный
        # допуск, который на маленьких весах его бы проглотил
        tol = np.maximum(np.abs(ref) * 2 ** -11, 63e-8)
        ok3 = bool((np.abs(g - ref) <= tol).all())
        n_el += ref.size
        n_dead_total[0] += n_dead
        n_blocks_total[0] += OUT * NB

        check(f"{key} [{OUT}x{IN}] bits={bits}", ok1 and ok2 and ok3,
              (f"вырожденных блоков {n_dead}" if n_dead else "")
              if ok1 and ok2 and ok3 else
              f"коды/ядро/значения = {ok1}/{ok2}/{ok3}, "
              f"max|Δ| {np.abs(g - ref).max():.3e}")

        if dump is not None:
            # в фикстуру идут первые DUMP_ROWS строк: раскладка от числа
            # строк не зависит вовсе (группы идут вдоль входной оси), а
            # emb/head целиком -- это 335 МБ на тензор, для эталона в
            # репозитории неприемлемо
            r = min(DUMP_ROWS, OUT)
            fixture[f"{key}::wq"] = mx.array(wq[:r])
            fixture[f"{key}::scales"] = mx.array(scales[:r])
            fixture[f"{key}::biases"] = mx.array(biases[:r])
            fixture[f"{key}::dense"] = mx.array(ref[:r]).astype(mx.bfloat16)
            fixture[f"{key}::bits"] = mx.array(np.array([bits], np.int32))

    print(f"\nпроверено {len(picked)} тензоров "
          f"({len(seen)} различных форм), {n_el / 1e6:.1f}M элементов")
    print(f"вырожденных блоков (scale ушёл под fp16): {n_dead_total[0]} из "
          f"{n_blocks_total[0]} ({100 * n_dead_total[0] / max(1, n_blocks_total[0]):.4f}%)"
          + ("  -- в контейнере у них scale=0 против 1e-8 в эталоне, "
             "расхождение <= 6.3e-7" if n_dead_total[0] else ""))

    if dump is not None:
        mx.save_safetensors(dump, fixture,
                            metadata={"note": "sb6 -> MLX affine, "
                                              "эталон для порта в Swift"})
        print(f"эталон -> {dump} ({os.path.getsize(dump) / 1e6:.1f} МБ, "
              f"{len(fixture)} буферов)")

    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else f"ПРОВАЛЕН: {len(FAILS)} тензоров"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
