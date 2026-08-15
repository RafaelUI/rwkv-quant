"""ГЕЙТ ПАМЯТИ ЗАГРУЗКИ: правки обязаны быть БИТ-В-БИТ, а не «примерно».

Правка трогает деквант (reader) и путь построения dense-параметров
(quant_model._dense). Обе -- чистая механика памяти, и ровно поэтому их
надо ловить равенством: изменение на последнем бите мантиссы здесь никак
себя не проявит на глаз, а в ppl уедет тихо (закон 15).

Четыре утверждения, все требуют РАВЕНСТВА:

  1. Новый `_dequantize_one` против ЗАМОРОЖЕННОЙ копии прежнего кода
     (broadcast+in-place против repeat_interleave) -- все три квантованные
     раскладки.
  2. `dequantize_banded` против `_dequantize_one(...).to(dtype)` при
     ПРИНУДИТЕЛЬНО мелкой полосе, то есть при многих полосах на тензор.
  3. Новый `_dense` против замороженной копии прежнего (включая dtype:
     правка решает fp16/fp32 ДО декванта, по qt.shape, а прежняя -- ПОСЛЕ,
     по форме результата; расхождение здесь означало бы, что часть
     параметров молча сменила точность).
  4. Полоса не вырождена: в чекпоинте обязан найтись тензор, который
     реально делится больше чем на одну полосу, -- иначе утверждение 2
     проверяет тождество (закон 17: синтетика, которую нельзя предъявить
     в реальном файле, не покрытие).

    python tests/test_dense_load_parity.py <model.rwkvq> [model.rwkvq ...]

Без аргументов берёт /tmp/reduction.rwkvq и /tmp/reduction_sym_head8.rwkvq.
ВНИМАНИЕ: гейт держит ОБА пути на одном тензоре, то есть заведомо дороже
по памяти, чем починенная загрузка. На 2.9B гонять по файлу на процесс.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.formats import reader  # noqa: E402
from rwkv_quant.formats.reader import (load_raw, _dequantize_one,  # noqa: E402
                                       dequantize_banded, can_band)
from rwkv_quant.formats.schema import (int8_codes, unpack6,  # noqa: E402
                                       unpack_nib_block, unpack_bitplane)


# --- ЗАМОРОЖЕННАЯ КОПИЯ ПРЕЖНЕГО КОДА (reader.py до правки памяти) ---------

def _old_sb6(qt):
    OUT, IN = qt.shape
    gs, NB = qt.gw_gs, IN // qt.gw_gs
    q = unpack_nib_block(qt.codes_packed, gs).to(torch.float32)
    if qt.gw_qh is not None:
        q = q + unpack_bitplane(qt.gw_qh, IN).to(torch.float32) * 16.0
    if qt.gw_qh2 is not None:
        q = q + unpack_bitplane(qt.gw_qh2, IN).to(torch.float32) * 32.0
    qs = unpack6(qt.gw_qsqm[..., :6], 8).reshape(OUT, NB).to(torch.float32)
    qm = (unpack6(qt.gw_qsqm[..., 6:], 8).reshape(OUT, NB).to(torch.int16)
          - 31).to(torch.float32)
    d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)
    dm = qt.gw_dm.float().repeat_interleave(qt.gw_sb, dim=1)
    scale = (qs * d).half().float().clamp_min(1e-8)
    mn = (qm * dm).half().float()
    return (q * scale.repeat_interleave(gs, dim=1)
            + mn.repeat_interleave(gs, dim=1)).to(torch.bfloat16)


def _old_sym(qt):
    OUT, IN = qt.shape
    gs = qt.gw_gs
    if qt.codes is not None:
        q = qt.codes.to(torch.float32)
    else:
        q = unpack_nib_block(qt.codes_packed, gs).to(torch.int16)
        if qt.gw_qh is not None:
            q = q + unpack_bitplane(qt.gw_qh, IN).to(torch.int16) * 16
        if qt.gw_qh2 is not None:
            q = q + unpack_bitplane(qt.gw_qh2, IN).to(torch.int16) * 32
        q = (q - 32).to(torch.float32)
    d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)
    scale = (qt.gw_qs.float() * d).half().float()
    return (q * scale.repeat_interleave(gs, dim=1)).to(torch.bfloat16)


def _old_asym(qt):
    OUT, IN = qt.shape
    gs = qt.gw_gs
    q = qt.codes.to(torch.float32)
    idx = torch.arange(IN) // gs
    return (q * qt.gw_scale[:, idx] + qt.gw_min[:, idx]).to(torch.bfloat16)


def _old_dequantize_one(qt):
    if qt.bits >= 16:
        return qt.dense
    if qt.gw_mode == "sb6":
        return _old_sb6(qt)
    if qt.gw_mode == "sym":
        return _old_sym(qt)
    if qt.gw_mode == "asym":
        return _old_asym(qt)
    w = (int8_codes(qt).float() * qt.scale.float()).to(torch.bfloat16)
    if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
        rows = qt.outlier_indices[:, 0].long()
        cols = qt.outlier_indices[:, 1].long()
        w[rows, cols] = qt.outlier_values
    return w


def _old_dense(qt):
    """quant_model._dense до правки."""
    t = _old_dequantize_one(qt) if qt.bits < 16 else qt.dense
    arr = mx.array(t.float().numpy())
    if arr.ndim == 2 and min(arr.shape) >= 32:
        return arr.astype(mx.float16)
    return arr


# --- новый _dense, взятый из продакшн-модуля -------------------------------

from rwkv_quant.backends.metal.quant_model import _dense  # noqa: E402


def _raw(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a)


def check_file(path, band_mb=1.0):
    ckpt = load_raw(path)
    n = {"sb6": 0, "sym": 0, "asym": 0, "rtn": 0, "dense": 0}
    n_banded = 0
    max_bands = 0
    fails = []

    for key, qt in ckpt.tensors.items():
        kind = qt.gw_mode if qt.gw_mode else ("dense" if qt.bits >= 16 else "rtn")
        n[kind] = n.get(kind, 0) + 1

        # 1. деквант: новый против замороженного прежнего
        if qt.bits < 16:
            new = _dequantize_one(qt)
            old = _old_dequantize_one(qt)
            if new.dtype != old.dtype or not torch.equal(new, old):
                fails.append(f"{key}: деквант разошёлся ({kind})")
                continue
            del new, old

        # 2. полосами против целиком, полоса ПРИНУДИТЕЛЬНО мелкая
        if can_band(qt):
            OUT, IN = qt.shape
            rows = max(1, int(band_mb * (1 << 20)) // (IN * 4))
            if rows < OUT:
                n_banded += 1
                max_bands = max(max_bands, -(-OUT // rows))
            for dt in (torch.bfloat16, torch.float16, torch.float32):
                got = dequantize_banded(qt, dt, chunk_mb=band_mb)
                want = _dequantize_one(qt).to(dt)
                if got.dtype != want.dtype or not torch.equal(got, want):
                    fails.append(f"{key}: полоса разошлась при {dt} ({kind})")
                del got, want

        # 3. _dense: новый против замороженного прежнего
        got = _raw(_dense(qt))
        want = _raw(_old_dense(qt))
        if got.dtype != want.dtype or got.shape != want.shape:
            fails.append(f"{key}: _dense сменил {want.dtype}/{want.shape} "
                         f"на {got.dtype}/{got.shape}")
        elif not np.array_equal(got, want):
            bad = int((got != want).sum())
            fails.append(f"{key}: _dense разошёлся в {bad} элементах ({kind})")
        del got, want

    return n, n_banded, max_bands, fails


def main():
    paths = sys.argv[1:] or ["/tmp/reduction.rwkvq",
                             "/tmp/reduction_sym_head8.rwkvq"]
    bad = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"ПРОПУЩЕН (нет файла): {path}")
            continue
        n, n_banded, max_bands, fails = check_file(path)
        comp = " ".join(f"{k}={v}" for k, v in n.items() if v)
        print(f"\n{os.path.basename(path)}: {comp}")
        print(f"  полосами реально резалось: {n_banded} тензоров, "
              f"максимум полос на тензор: {max_bands}")
        if max_bands < 2:
            print("  ПРОВАЛ: ни один тензор не разбился больше чем на одну "
                  "полосу -- утверждение 2 вырождено (закон 17)")
            bad += 1
        if fails:
            bad += len(fails)
            for f in fails[:20]:
                print(f"  ПРОВАЛ {f}")
            if len(fails) > 20:
                print(f"  ... и ещё {len(fails)-20}")
        else:
            print("  равенство: ЗЕЛЁНЫЙ (деквант, полосы, _dense)")

    print()
    if bad:
        print(f"ГЕЙТ КРАСНЫЙ: {bad} расхождений")
        sys.exit(1)
    print("ГЕЙТ ЗЕЛЁНЫЙ")


if __name__ == "__main__":
    main()
