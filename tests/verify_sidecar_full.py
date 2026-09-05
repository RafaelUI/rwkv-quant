"""
Гейт полного сайдкара: восстановление из *.safetensors+json БЕЗ torch
совпадает с эталонным деквантом .rwkvq.

Это одновременно и проверка, и РЕФЕРЕНСНАЯ РЕАЛИЗАЦИЯ читалки для
rwkv-metal и SwiftRWKV: функция dequant_from_sidecar ниже покрывает все
четыре раскладки (sb6 / asym / rtn / dense) и не импортирует torch.
Torch здесь нужен только эталону, с которым сверяемся.

Ориентацию `dequant_from_sidecar` НЕ применяет и применять не должна:
деквант идёт по тем осям, вдоль которых считались блоки и scale.
Потребителю нужен `linear_weight_from_sidecar` -- он же показывает,
ОТКУДА берётся ориентация: поле `transposed` в манифесте, а при его
отсутствии (сайдкары старше 05.09) -- `codec.is_transposed`, то есть
таблица имён. Порт на другой язык обязан повторить ОБЕ ветки: без первой
он зашьёт таблицу имён у себя, без второй перестанет читать уже
выкаченные файлы.

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
from rwkv_quant.formats import codec  # noqa: E402

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

    if kind == "sym":
        # Q6_K: min нет, поэтому нет и клипа scale снизу -- у вырожденного
        # блока и масштаб, и коды нулевые. bits различается ПО БУФЕРАМ:
        # при восьми битах qblk -- знаковые байты, при шести -- ниббл со
        # сдвигом +32 плюс две битплоскости.
        gs, sb, bits = meta["gw_gs"], meta["gw_sb"], meta["bits"]
        NB, NSB = IN // gs, IN // (gs * sb)
        blk = arrays[f"{key}::qblk"]
        if bits == 8:
            q = mx.view(blk, mx.int8).reshape(OUT, NB, gs).astype(mx.float32)
        else:
            NP = IN // 32
            b = blk.reshape(OUT, NP, 24)
            cb = b[:, :, :16].reshape(OUT, NP * 2, 8)
            q = mx.concatenate([cb & 0xF, cb >> 4], axis=2).astype(mx.float32)
            for i in range(2):
                plane = b[:, :, 16 + 4 * i:20 + 4 * i].reshape(OUT, IN // 8)
                q = q + _unpack_bits(plane, IN, i).reshape(
                    OUT, NB, gs).astype(mx.float32) * (16.0 * (2 ** i))
            q = q - 32.0
        qs = mx.view(arrays[f"{key}::qs"], mx.int8).astype(mx.float32)
        d = mx.repeat(arrays[f"{key}::d"].astype(mx.float32), NB // NSB, axis=1)
        scale = (qs * d).astype(mx.float16).astype(mx.float32)
        return (q * scale.reshape(OUT, NB, 1)).reshape(OUT, IN).astype(mx.bfloat16)

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


def linear_weight_from_sidecar(arrays, manifest, key) -> mx.array:
    """Вес в конвенции nn.Linear [out, in] -- то, что нужно ПОТРЕБИТЕЛЮ."""
    # dequant_from_sidecar возвращает вес в той раскладке, в какой он ЛЕЖИТ,
    # и это правильно: деквант обязан идти по тем осям, вдоль которых
    # считались блоки и scale. Ориентация применяется ПОСЛЕ, и берётся ИЗ
    # МАНИФЕСТА, а не из зашитой у себя таблицы имён: её зашитость стоила
    # rwkv-metal падения на композиции весов (0.3.2) и до сих пор живёт
    # восемью .transposed() в RwkvqFullConvert.swift.
    #
    # Отсутствие поля transposed -- не ошибка, а сайдкар старше 05.09:
    # codec.is_transposed падает в таблицу имён и даёт прежнее поведение.
    # ПОРТ НА ДРУГОЙ ЯЗЫК ОБЯЗАН СОХРАНИТЬ этот запасной путь, иначе уже
    # выкаченные сайдкары перестанут читаться.
    w = dequant_from_sidecar(arrays, key, manifest["tensors"][key])
    if not codec.is_transposed(manifest, key):
        return w
    assert w.ndim == 2, "%s: transposed на тензоре ndim=%d" % (key, w.ndim)
    return w.T


LORA_SUF = ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")


def check_orientation(manifest, arrays, ckpt) -> int:
    """Ориентация доехала, совпала с источником и складывается по формам."""
    bad = 0
    tt = manifest["tensors"]
    nof = [k for k, m in tt.items() if "transposed" not in m]
    if nof:
        print("  !! поля transposed нет у %d тензоров (сайдкар старше 05.09, "
              "переэкспортировать): %s" % (len(nof), nof[:3]))
        bad += 1
    else:
        print("поле transposed есть у всех %d тензоров" % len(tt))

    diff = [k for k, m in tt.items() if "transposed" in m
            and bool(m["transposed"]) != bool(ckpt.tensors[k].transposed)]
    if diff:
        print("  !! ориентация расходится с .rwkvq у %d: %s" % (len(diff), diff[:3]))
        bad += 1
    else:
        # Сравнивать можно только там, где поле есть. Печатать
        # «совпадает» по пустому множеству -- то самое зелёное утверждение
        # ни о чём, из-за которого написан закон 31.
        have = [k for k, m in tt.items() if "transposed" in m]
        n_tr = sum(1 for m in tt.values() if m.get("transposed"))
        if have:
            print("ориентация совпадает с .rwkvq на %d ключах с полем из %d "
                  "(транспонированных %d)" % (len(have), len(tt), n_tr))
        else:
            print("  сверка значений НЕ ПРОВОДИЛАСЬ: поля нет ни у одного ключа")

    # КОМПОЗИЦИЯ. Низкоранговые в Linear-конвенции обязаны встать к n_embd
    # нужной стороной: *1 это [rank, D], *2 это [D, rank]. Ранги (96/64/256)
    # не равны D, поэтому неверный флаг ставит D не на ту ось и ловится,
    # а не проходит незаметно. Это та же проверка, на которой падал
    # rwkv-metal, только сделанная заранее и на формах.
    D = manifest["n_embd"]
    checked = 0
    for key in tt:
        base = key.rsplit(".", 1)[-1]
        if base not in LORA_SUF or not key.startswith("blocks.0."):
            continue
        w = linear_weight_from_sidecar(arrays, manifest, key)
        want = 1 if base.endswith("1") else 0
        checked += 1
        if w.shape[want] != D:
            print("  !! %s: Linear-форма %s, n_embd=%d ожидался на оси %d"
                  % (key, tuple(w.shape), D, want))
            bad += 1
    if checked == 0:
        print("  !! низкоранговых ключей не нашлось -- проверка композиции "
              "ничего не значит")
        bad += 1
    else:
        print("композиция Linear-формы сошлась на %d низкоранговых" % checked)
    return bad


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

    seen, bad = {}, check_orientation(manifest, arrays, ckpt)
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
