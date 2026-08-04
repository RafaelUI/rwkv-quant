"""
Гейт шага 4: контейнер .rwkvq переехал с pickle на safetensors.

Что доказывается:
  1. Роундтрип без потерь: каждое поле каждого QuantizedTensor после
     save_rwkvq -> load_raw совпадает по dtype, форме и БИТАМ, метаданные
     чекпоинта тоже.
  2. Прежние файлы читаются: load_raw различает контейнеры по первым
     байтам, старый pickle грузится как раньше.
  3. Деквант не сдвинулся: reader на новом контейнере даёт ровно то же,
     что на старом.
  4. TORCH-FREE ПУТЬ РАБОТАЕТ: codec.open_rwkvq + codec.dequant_key,
     ноль импортов torch внутри, совпадает с reader бит-в-бит. Это и есть
     то, ради чего менялся контейнер -- порт в rwkv-metal/SwiftRWKV.
  5. Негативные контроли: safetensors без манифеста и манифест из
     будущего должны падать, а не молча возвращать мусор.

    python tests/test_rwkvq_container.py <legacy_или_новый.rwkvq> [out.rwkvq]

Вход берётся готовым файлом, а не квантованием на месте: гейт про
контейнер, а не про квантование, и на 2.9B пересчёт стоил бы минуты.
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.formats import codec  # noqa: E402
from rwkv_quant.formats.reader import load_raw, _dequantize_one  # noqa: E402
from rwkv_quant.formats.writer import TENSOR_FIELDS, save_rwkvq  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def test_roundtrip(src, out):
    print(f"роундтрип: {src} -> {out}")
    src_is_zip = open(src, "rb").read(4) == codec.MAGIC_ZIP
    print(f"  исходный контейнер: {'pickle (zip)' if src_is_zip else 'safetensors'}")

    old = load_raw(src)
    save_rwkvq(old, out)
    new = load_raw(out)

    for f in ("naming", "n_layer", "n_embd", "head_size", "vocab_size",
              "config_repr"):
        check(f"метаданные: {f}", getattr(old, f) == getattr(new, f),
              f"{getattr(old, f)!r} против {getattr(new, f)!r}")
    check("состав тензоров", set(old.tensors) == set(new.tensors),
          f"{len(old.tensors)} против {len(new.tensors)}")

    bad_fields, bad_scalars, n_buf = [], [], 0
    for key, a in old.tensors.items():
        b = new.tensors[key]
        for s in ("bits", "group", "gw_mode", "gw_gs", "gw_sb"):
            if getattr(a, s) != getattr(b, s):
                bad_scalars.append(f"{key}.{s}")
        if tuple(a.shape) != tuple(b.shape):
            bad_scalars.append(f"{key}.shape")
        for f in TENSOR_FIELDS:
            va, vb = getattr(a, f), getattr(b, f)
            empty_a = va is None or va.numel() == 0
            if empty_a != (vb is None or vb.numel() == 0):
                bad_fields.append(f"{key}::{f} (наличие)")
                continue
            if empty_a:
                continue
            n_buf += 1
            if va.dtype != vb.dtype or va.shape != vb.shape:
                bad_fields.append(f"{key}::{f} ({va.dtype}/{va.shape} против "
                                  f"{vb.dtype}/{vb.shape})")
            elif not bool((va == vb).all()):
                bad_fields.append(f"{key}::{f} (биты)")
    check("скаляры тензоров", not bad_scalars, ", ".join(bad_scalars[:3]))
    check(f"буферы бит-в-бит ({n_buf} шт.)", not bad_fields,
          ", ".join(bad_fields[:3]))

    so, sn = os.path.getsize(src), os.path.getsize(out)
    print(f"  размер: {so / 1e6:.1f} МБ -> {sn / 1e6:.1f} МБ "
          f"({100 * (sn - so) / so:+.2f}%)")
    return old, new


def test_dequant(old, new):
    print("\nдеквант через reader: новый контейнер против старого")
    bad = 0
    for key, a in old.tensors.items():
        wa, wb = _dequantize_one(a), _dequantize_one(new.tensors[key])
        if wa.shape != wb.shape or not bool((wa == wb).all()):
            bad += 1
            if bad <= 3:
                print(f"  !! {key}")
        del wa, wb
    check(f"деквант совпал ({len(old.tensors)} тензоров)", bad == 0,
          "" if not bad else f"{bad} расхождений")


def test_torch_free(out, new):
    print("\ntorch-free путь: codec.open_rwkvq + dequant_key против reader")
    manifest, arrays = codec.open_rwkvq(out)
    check("манифест: формат и версия",
          manifest["format"] == codec.FORMAT
          and manifest["format_version"] == codec.FORMAT_VERSION,
          f"{manifest['format']!r} v{manifest['format_version']}")
    check("манифест: состав", set(manifest["tensors"]) == set(new.tensors))

    kinds, bad = {}, 0
    for key, meta in manifest["tensors"].items():
        got = codec.dequant_key(manifest, arrays, key)
        ref = _dequantize_one(new.tensors[key])
        # codec отдаёт float32, reader -- bf16; сводим к bf16 (для dense
        # это тождество: он и хранится в bf16)
        got_bf = torch.from_numpy(np.ascontiguousarray(got)).to(torch.bfloat16)
        ok = got_bf.shape == ref.shape and bool((got_bf == ref).all())
        k = meta["kind"]
        kinds.setdefault(k, [0, 0])
        kinds[k][0] += 1
        if not ok:
            kinds[k][1] += 1
            bad += 1
            if bad <= 3:
                print(f"  !! {key} ({k})")
        del got, got_bf, ref
    for k, (total, nb) in sorted(kinds.items()):
        check(f"torch-free {k}", nb == 0,
              f"{total} тензоров" if not nb else f"{nb} из {total} разошлись")


def test_negative(out, tmpdir):
    print("\nнегативные контроли")

    from safetensors.torch import save_file
    plain = os.path.join(tmpdir, "_no_manifest.safetensors")
    save_file({"x": torch.zeros(4)}, plain)
    try:
        load_raw(plain)
        check("safetensors без манифеста отвергается", False, "загрузился")
    except ValueError:
        check("safetensors без манифеста отвергается", True)

    # манифест из будущего: правим только версию, байты данных не трогаем
    future = os.path.join(tmpdir, "_future.rwkvq")
    with open(out, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        rest = f.read()
    m = json.loads(header["__metadata__"]["rwkvq"])
    m["format_version"] = codec.FORMAT_VERSION + 1
    header["__metadata__"]["rwkvq"] = json.dumps(m)
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)          # safetensors требует выравнивания
    with open(future, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + rest)
    try:
        load_raw(future)
        check("format_version из будущего отвергается", False, "загрузился")
    except ValueError:
        check("format_version из будущего отвергается", True)
    for p in (plain, future):
        os.remove(p)


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src + ".st_roundtrip"
    old, new = test_roundtrip(src, out)
    test_dequant(old, new)
    del old
    test_torch_free(out, new)
    test_negative(out, os.path.dirname(out) or ".")
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else f"ПРОВАЛЕН: {', '.join(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
