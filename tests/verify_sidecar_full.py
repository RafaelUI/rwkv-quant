"""
Гейт полного сайдкара: восстановление из *.safetensors+json БЕЗ torch
совпадает с эталонным деквантом .rwkvq.

Это одновременно и проверка, и РЕФЕРЕНСНАЯ РЕАЛИЗАЦИЯ читалки для
rwkv-metal и SwiftRWKV: функция dequant_from_sidecar ниже покрывает все
четыре раскладки (sb6 / asym / rtn / dense) и не импортирует torch.
Torch здесь нужен только эталону, с которым сверяемся.

Порог: сравниваем в bf16. Пути арифметически одинаковы, но порядок
операций у MLX и torch различается, поэтому гейт -- не бит-в-бит, а
"расхождение не больше одного ulp bf16 на элемент". Доля точных
совпадений печатается отдельно: если она резко проседает на какой-то
раскладке, это баг, а не округление.

    python tests/verify_sidecar_full.py <sidecar_без_расширения> <model.rwkvq>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.formats.reader import load_raw, _dequantize_one  # noqa: E402

# сколько тензоров каждой раскладки проверять (emb/head крупные, полный
# перебор 1062 тензоров занял бы минуты без прибавки к доверию)
PER_KIND = 6


def _unpack_bits(byte_arr, out_cols, shift):
    """Битплоскость uint8 [OUT, IN/8] -> {0,1} [OUT, out_cols]."""
    bits = (byte_arr[..., None] >> mx.arange(8, dtype=mx.uint8)) & 1
    return bits.reshape(byte_arr.shape[0], -1)[:, :out_cols]


def dequant_from_sidecar(arrays, key, meta) -> mx.array:
    """ЕДИНСТВЕННАЯ точка восстановления веса из сайдкара. torch-free."""
    kind = meta["kind"]

    if kind == "dense":
        # форма произвольная: нормы 1-D, token-shift миксы (1,1,C),
        # bias-термы LoRA-веток -- квантованию не подлежат вовсе
        return arrays[f"{key}::dense"]

    if kind == "rtn" and not meta["packed"]:
        # per-row RTN не требует знания формы: scale имеет вид [d0,1,...]
        # и вещается сам. Формы бывают и 3-D -- writer квантует всё, что
        # dim >= 2, включая (1,1,C)-параметры, попавшие в группу.
        w = (arrays[f"{key}::codes"].astype(mx.float32)
             * arrays[f"{key}::scale"].astype(mx.float32))
        if f"{key}::outlier_idx" in arrays:
            oi = np.array(arrays[f"{key}::outlier_idx"])
            ov = np.array(arrays[f"{key}::outlier_val"].astype(mx.float32))
            wn = np.array(w)
            wn[oi[:, 0], oi[:, 1]] = ov
            w = mx.array(wn)
        return w.astype(mx.bfloat16)

    OUT, IN = meta["shape"]        # остальные раскладки только 2-D

    if kind == "sb6":
        gs, sb, xbits = meta["gw_gs"], meta["gw_sb"], meta["xbits"]
        NB, NSB = IN // gs, IN // (gs * sb)
        blk = arrays[f"{key}::qblk"].reshape(OUT, NB, 16 + 4 * xbits)
        cb = blk[:, :, :16]
        q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.float32)
        for i in range(xbits):     # битплоскости 5-го и 6-го бита
            plane = blk[:, :, 16 + 4 * i:20 + 4 * i].reshape(OUT, IN // 8)
            q = q + _unpack_bits(plane, IN, i).reshape(
                OUT, NB, gs).astype(mx.float32) * (16.0 * (2 ** i))
        sm = arrays[f"{key}::qsqm"].reshape(OUT, NB, 2)
        qs = sm[:, :, 0].astype(mx.float32)
        # qm уже со снятым сдвигом -31 на этапе экспорта (см. GwQuantLinear),
        # здесь только реинтерпретация байта как знакового
        qm = mx.view(sm[:, :, 1], mx.int8).astype(mx.float32)
        dd = arrays[f"{key}::ddm"].reshape(OUT, NSB, 2)
        d = mx.repeat(dd[:, :, 0].astype(mx.float32), NB // NSB, axis=1)
        dm = mx.repeat(dd[:, :, 1].astype(mx.float32), NB // NSB, axis=1)
        # half-роундтрип обязателен: кернель и writer считают именно так
        scale = mx.maximum((qs * d).astype(mx.float16).astype(mx.float32), 1e-8)
        mn = (qm * dm).astype(mx.float16).astype(mx.float32)
        w = q * scale.reshape(OUT, NB, 1) + mn.reshape(OUT, NB, 1)
        return w.reshape(OUT, IN).astype(mx.bfloat16)

    if kind == "asym":
        gs = meta["gw_gs"]
        q = arrays[f"{key}::codes"].astype(mx.float32)
        idx = mx.arange(IN) // gs
        w = (q * arrays[f"{key}::gw_scale"][:, idx]
             + arrays[f"{key}::gw_min"][:, idx])
        return w.astype(mx.bfloat16)

    if kind == "rtn":
        # сюда доходит только упакованный вариант (bits <= 4): biased
        # split-нибблы, байт i несёт колонку i в low и i + ceil(IN/2) в
        # high, в ниббле лежит code + 8
        p = arrays[f"{key}::codes_packed"]
        lo = (p & 0xF).astype(mx.int16) - 8
        hi = (p >> 4).astype(mx.int16) - 8
        q = mx.concatenate([lo, hi], axis=1)[:, :IN].astype(mx.float32)
        w = q * arrays[f"{key}::scale"].astype(mx.float32)
        if f"{key}::outlier_idx" in arrays:
            oi = np.array(arrays[f"{key}::outlier_idx"])
            ov = np.array(arrays[f"{key}::outlier_val"].astype(mx.float32))
            wn = np.array(w)
            wn[oi[:, 0], oi[:, 1]] = ov
            w = mx.array(wn)
        return w.astype(mx.bfloat16)

    raise ValueError(f"{key}: неизвестная раскладка {kind}")


def main():
    sidecar, rwkvq = sys.argv[1], sys.argv[2]
    manifest = json.load(open(sidecar + ".json"))
    arrays = mx.load(sidecar + ".safetensors")
    print(f"манифест v{manifest.get('format_version')}, "
          f"{len(manifest['tensors'])} тензоров, буферов {len(arrays)}")

    ckpt = load_raw(rwkvq)
    assert set(manifest["tensors"]) == set(ckpt.tensors), \
        "состав тензоров сайдкара и .rwkvq расходится"
    for f in ("naming", "n_layer", "n_embd", "head_size", "vocab_size"):
        assert manifest[f] == getattr(ckpt, f), f"метаданные: {f}"
    print("состав и метаданные совпадают")

    seen, bad = {}, 0
    for key, meta in manifest["tensors"].items():
        kind = meta["kind"]
        seen.setdefault(kind, [])
        if len(seen[kind]) >= PER_KIND:
            continue
        got = np.array(dequant_from_sidecar(arrays, key, meta)
                       .astype(mx.float32))
        ref = _dequantize_one(ckpt.tensors[key]).float().numpy()
        if got.shape != ref.shape:
            print(f"  !! {key}: форма {got.shape} против {ref.shape}")
            bad += 1
            continue
        exact = float((got == ref).mean())
        # один ulp bf16 -- это 2^-8 относительной точности
        tol = np.maximum(np.abs(ref) * 2 ** -8, 1e-7)
        within = float((np.abs(got - ref) <= tol).mean())
        seen[kind].append((key, exact, within, float(np.abs(got - ref).max())))
        if within < 1.0:
            bad += 1
            print(f"  !! {key} ({kind}): в допуске лишь {100*within:.4f}%")

    print()
    for kind, rows in sorted(seen.items()):
        ex = np.mean([r[1] for r in rows])
        wi = np.mean([r[2] for r in rows])
        mx_ = max(r[3] for r in rows)
        n_all = sum(1 for m in manifest["tensors"].values() if m["kind"] == kind)
        print(f"{kind:<7} проверено {len(rows)}/{n_all}: "
              f"точных {100*ex:7.3f}%, в допуске {100*wi:7.3f}%, "
              f"max|Δ| {mx_:.3e}")

    print("\nГЕЙТ " + ("ПРОЙДЕН" if bad == 0 else f"ПРОВАЛЕН ({bad})"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
