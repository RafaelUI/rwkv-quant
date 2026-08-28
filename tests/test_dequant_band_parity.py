"""
Гейт: деквант ПОЛОСАМИ строк побитово равен декванту целиком.

Ссылка на этот файл стояла в двух местах (`reader.DEQUANT_CHUNK_MB` и
`reader._RowBand`) с 15.08, а самого файла не было -- то есть нарезка,
снимающая гигабайты пика, держалась на честном слове докстринга. Это тот
же класс дыры, что закон 31 (гейт, который ничего не проверил, обязан
краснеть), только в предельной форме.

Проверяются ОБЕ реализации, потому что их две и они независимы:

  A. numpy, torch-free: codec.dequant_key(chunk_mb=малый) против
     codec.dequant_key(chunk_mb=0). Это путь QLoRA-загрузчика rwkv-metal
     (model/convert.py:220) и порта в SwiftRWKV.
  B. torch: reader.dequantize_banded против reader._dequantize_one.

Равенство требуется ПОБИТОВОЕ, а не с допуском: нарезка обязана быть
перекладкой, а не другой схемой (закон 15). Сравнение идёт по битовому
виду, чтобы отличить -0.0 от 0.0 и поймать NaN.

Полоса берётся заведомо мелкая (по умолчанию 1 МБ), чтобы у крупных
тензоров получилось МНОГО полос, а у мелких сработал бы возврат на
нерезаный путь -- обе ветки должны быть пройдены, и это печатается.

    python tests/test_dequant_band_parity.py [model.rwkvq ...] [--all]

Без аргументов берутся файлы, которые кладёт tests/restore_tmp.sh. Если
не проверено НИ ОДНОГО тензора -- гейт красный (закон 31).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from rwkv_quant.formats import codec  # noqa: E402

DEFAULT_PATHS = ["/tmp/reduction_new.rwkvq", "/tmp/champion_v2.rwkvq"]
CHUNK_MB = float(os.environ.get("BAND_GATE_CHUNK_MB", "1"))

FAILS = []
N_CHECKED = 0
N_BANDED = 0
N_WHOLE = 0


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def bitwise_equal(a, b) -> bool:
    """Побитовое равенство двух float32-массивов."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def pick_keys(manifest, want_all):
    """Все ключи при --all, иначе по одному представителю на пару
    (kind, group) ПЛЮС всё, что крупнее 16 МБ: emb и head обязаны быть
    проверены всегда -- ради них нарезка и делалась."""
    keys = list(manifest["tensors"].keys())
    if want_all:
        return keys
    seen, out = set(), []
    for k in keys:
        m = manifest["tensors"][k]
        shape = tuple(m["shape"])
        n = 1
        for d in shape:
            n *= d
        tag = (m["kind"], m.get("group", ""), len(shape))
        if n * 4 >= (16 << 20) or tag not in seen:
            seen.add(tag)
            out.append(k)
    return out


def test_numpy(path, want_all):
    global N_CHECKED, N_BANDED, N_WHOLE
    print(f"\nA. numpy codec.dequant_key -- {path}")
    manifest, arrays = codec.open_rwkvq(path)
    for key in pick_keys(manifest, want_all):
        m = manifest["tensors"][key]
        shape = tuple(m["shape"])
        whole = codec.dequant_key(manifest, arrays, key, chunk_mb=0)
        band = codec.dequant_key(manifest, arrays, key, chunk_mb=CHUNK_MB)
        ok = bitwise_equal(whole, band)
        N_CHECKED += 1
        nb = 0
        if codec.can_band(manifest, key) and len(shape) == 2:
            rows = max(1, int(CHUNK_MB * (1 << 20)) // max(1, shape[1] * 4))
            nb = -(-shape[0] // rows) if rows < shape[0] else 1
        if nb > 1:
            N_BANDED += 1
        else:
            N_WHOLE += 1
        check(f"{key} [{m['kind']}] {shape} полос={nb or 1}", ok,
              "" if ok else "РАСХОЖДЕНИЕ полос и целого")
        del whole, band


def test_torch(path, want_all):
    global N_CHECKED
    try:
        import torch
        from rwkv_quant.formats import reader
    except Exception as e:
        check(f"torch-путь на {path}", False, f"не импортируется: {e}")
        return
    print(f"\nB. torch reader.dequantize_banded -- {path}")
    ckpt = reader.load_raw(path)
    manifest = {"tensors": {}}
    keys = list(ckpt.tensors.keys())
    if not want_all:
        seen, sel = set(), []
        for k in keys:
            qt = ckpt.tensors[k]
            n = 1
            for d in tuple(qt.shape):
                n *= d
            tag = (getattr(qt, "gw_mode", ""), qt.bits, len(tuple(qt.shape)))
            if n * 4 >= (16 << 20) or tag not in seen:
                seen.add(tag)
                sel.append(k)
        keys = sel
    del manifest
    for key in keys:
        qt = ckpt.tensors[key]
        whole = reader._dequantize_one(qt).to(torch.float32)
        band = reader.dequantize_banded(qt, torch.float32, chunk_mb=CHUNK_MB)
        ok = bool(torch.equal(whole.view(torch.int32), band.view(torch.int32)))
        N_CHECKED += 1
        check(f"{key} bits={qt.bits} {tuple(qt.shape)} "
              f"band={reader.can_band(qt)}", ok,
              "" if ok else "РАСХОЖДЕНИЕ полос и целого")
        del whole, band


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    want_all = "--all" in sys.argv[1:]
    paths = [p for p in (args or DEFAULT_PATHS) if os.path.exists(p)]
    if not paths:
        print("нет ни одного .rwkvq: ждал " + ", ".join(args or DEFAULT_PATHS))
        print("  (tests/restore_tmp.sh кладёт умолчания)")
        FAILS.append("нечего проверять")
    for p in paths:
        test_numpy(p, want_all)
        test_torch(p, want_all)
    print(f"\nпроверено тензоров: {N_CHECKED} "
          f"(numpy: {N_BANDED} нарезанных, {N_WHOLE} нерезаных)")
    if N_CHECKED == 0:
        FAILS.append("ни одного проверенного случая")
    if N_BANDED == 0:
        FAILS.append("ни одного НАРЕЗАННОГО случая -- гейт ничего не доказал")
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else f"ПРОВАЛЕН: {', '.join(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
