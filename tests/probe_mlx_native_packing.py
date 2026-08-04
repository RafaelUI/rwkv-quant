"""
Раскладка mx.quantize: одна ли она для 4, 5, 6 и 8 бит.

Зачем. Родное ядро MLX (`quantized_matmul`) на формах 2.9B быстрее
нынешнего пути SwiftRWKV в 2.1-4.9x и быстрее плотного bf16 в 1.3-2.1x
(замер SwiftRWKV/decode-bench --micro). Чтобы им пользоваться, наши
sb6-коды надо разложить в контейнер MLX БЕЗ пересчёта чисел:
mx.quantize(dense) пересчитывает scale/bias по min/max блока и рушит
калибровку (~89% значений, tests/dev_check_requantize_roundtrip.py в
rwkv-metal). В rwkv-metal раскладка выведена one-hot тестами ТОЛЬКО для
bits=6 (lora/rwkvq_native.py), а COMPRESSION живёт на 5 и 4 битах.

Метод. Реверс-инжиниринг здесь не нужен вовсе: `mx.dequantize` --
обратная функция к упаковке, так что свой упаковщик проверяется прямо
против неё. Гипотеза (из bits=6): группа из 32 кодов пишется LSB-first
непрерывным битовым потоком, поле позиции p начинается на глобальном
бите p*bits и переходит границу 32-битного слова без выравнивания.
Проверяется так: пакуем СЛУЧАЙНЫЕ коды, даём MLX распаковать и сверяем
с codes*scale + bias. Совпало на всех формах и битностях -- правило то
же; разошлось -- у этой битности своя ветка (у степеней двойки в
quantized.h она отдельная, см. get_bytes_per_pack).

    python tests/probe_mlx_native_packing.py
"""
import sys

import mlx.core as mx
import numpy as np

GROUP = 32
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def pack_codes(codes32: np.ndarray, bits: int) -> np.ndarray:
    """[..., 32] целых 0..2^bits-1 -> [..., bits] uint32.

    Обобщение _pack_codes_mlx6 на произвольную битность: на группу из 32
    кодов уходит ровно 32*bits бит = bits слов. Поле p начинается на
    глобальном бите p*bits и может пересечь границу слова."""
    lead = codes32.shape[:-1]
    words = np.zeros((*lead, bits), dtype=np.uint32)
    c = codes32.astype(np.uint32)
    for p in range(GROUP):
        start = p * bits
        w0, off = start // 32, start % 32
        lo_bits = min(bits, 32 - off)
        hi_bits = bits - lo_bits
        words[..., w0] |= ((c[..., p] & ((1 << lo_bits) - 1)) << off).astype(np.uint32)
        if hi_bits:
            words[..., w0 + 1] |= ((c[..., p] >> lo_bits)
                                   & ((1 << hi_bits) - 1)).astype(np.uint32)
    return words


def probe(bits: int, OUT: int = 8, NB: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed + bits)
    qmax = (1 << bits) - 1
    codes = rng.integers(0, qmax + 1, (OUT, NB, GROUP)).astype(np.uint32)
    scales = rng.uniform(1e-3, 1e-2, (OUT, NB)).astype(np.float16)
    biases = rng.uniform(-1e-2, 1e-2, (OUT, NB)).astype(np.float16)

    wq = mx.array(pack_codes(codes, bits).reshape(OUT, NB * bits))
    got = mx.dequantize(wq, mx.array(scales), mx.array(biases),
                        group_size=GROUP, bits=bits)
    mx.eval(got)

    # СРАВНИВАЮТСЯ КОДЫ, А НЕ ЗНАЧЕНИЯ. Первая версия сверяла
    # w = q*scale + bias в fp16 и давала 75-81% совпадений на ВСЕХ
    # битностях, включая ту, что в rwkv-metal проверена бит-в-бит. Это
    # был не разбор раскладки, а разное округление: порядок операций и
    # точность накопления у ядра свои. Обратный ход (got - bias)/scale с
    # округлением возвращает целое, и оно либо то, либо не то.
    g = np.array(got.astype(mx.float32)).reshape(OUT, NB, GROUP)
    back = np.rint((g - biases[..., None].astype(np.float32))
                   / scales[..., None].astype(np.float32)).astype(np.int64)
    exact = float((back == codes.astype(np.int64)).mean())
    check(f"bits={bits}: раскладка та же", exact == 1.0,
          "" if exact == 1.0 else f"совпало {100*exact:.2f}% кодов")
    return exact == 1.0


def probe_roundtrip(bits: int):
    """Контроль в обратную сторону: то, что упаковал сам MLX, наш
    распаковщик должен прочитать теми же кодами. Иначе можно было бы
    пройти проверку выше на упаковщике, который «согласован сам с собой»
    по случайному совпадению маскирования."""
    OUT, NB = 4, 3
    rng = np.random.default_rng(100 + bits)
    w = mx.array(rng.normal(0, 0.02, (OUT, NB * GROUP)).astype(np.float32))
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=bits)
    mx.eval(wq, sc, bi)

    # распаковка нашим правилом
    words = np.array(wq).reshape(OUT, NB, bits)
    codes = np.zeros((OUT, NB, GROUP), dtype=np.uint32)
    for p in range(GROUP):
        start = p * bits
        w0, off = start // 32, start % 32
        lo_bits = min(bits, 32 - off)
        hi_bits = bits - lo_bits
        v = (words[..., w0] >> off) & ((1 << lo_bits) - 1)
        if hi_bits:
            v = v | ((words[..., w0 + 1] & ((1 << hi_bits) - 1)) << lo_bits)
        codes[..., p] = v

    # эталонные коды -- те, что вернёт само ядро через dequantize
    ref = np.array(mx.dequantize(wq, sc, bi, group_size=GROUP, bits=bits)
                   .astype(mx.float32)).reshape(OUT, NB, GROUP)
    s = np.array(sc).astype(np.float32)[..., None]
    b = np.array(bi).astype(np.float32)[..., None]
    ref_codes = np.rint((ref - b) / s).astype(np.int64)
    exact = float((codes.astype(np.int64) == ref_codes).mean())
    check(f"bits={bits}: обратное чтение", exact == 1.0,
          "" if exact == 1.0 else f"совпало {100*exact:.2f}% кодов")


def main():
    print("раскладка mx.quantize, группа 32\n")
    print("прямая проверка: наша упаковка -> mx.dequantize")
    ok = {b: probe(b) for b in (4, 5, 6, 8)}
    print("\nобратная проверка: mx.quantize -> наша распаковка")
    for b in (4, 5, 6, 8):
        probe_roundtrip(b)

    print("\nитог по пресетам:")
    print(f"  REDUCTION   (proj/cmix/emb_head @6)   -> "
          f"{'ГОДИТСЯ' if ok.get(6) else 'НЕ ГОДИТСЯ'}")
    print(f"  COMPRESSION (proj@5, cmix@4, head@5) -> "
          f"{'ГОДИТСЯ' if ok.get(5) and ok.get(4) else 'НЕ ГОДИТСЯ'}")
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else f"ПРОВАЛЕН: {', '.join(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
