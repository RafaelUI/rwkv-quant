"""
Гейт шага 3: torch-free codec.py бит-в-бит равен torch-пути.

Проверяются два независимых утверждения.

  A. Упаковщики. codec.* против schema.* -- две независимые реализации
     одной раскладки (дублирование осознанное, цена единого источника
     измерена в bench_codec_dequant_ab.py: 1.65-2.63x). Torch-версии
     продублированы сюда ВЕРБАТИМ как _ref_*, чтобы гейт ловил и правку
     schema.py тоже, а не только расхождение codec с текущим schema.

  B. Деквант. codec.dequant_* -- нормативная torch-free реализация для
     порта в rwkv-metal / SwiftRWKV, reader._dequantize_one -- быстрая
     torch'евая. Сверяются оба, плюс третий путь _ref_dequantize_one
     (деквант целиком на прежнем torch-коде): он доказывает, что правки
     в schema.py не сдвинули выход reader'а ни на бит.

Уровни:
  1. роундтрип pack -> unpack на каждом упаковщике;
  2. numpy против прежнего torch на случайных данных, все формы и
     граничные случаи (нечётное число колонок, gs 32/64, xbits 0/1/2);
  3. три пути декванта на синтетических QuantizedTensor всех четырёх
     раскладок -- в bf16, БЕЗ допуска, ожидается точное равенство;
  4. то же на настоящем чекпоинте, если путь передан аргументом.

    python tests/test_codec_parity.py [model.rwkvq ...]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.formats import codec, schema  # noqa: E402
from rwkv_quant.formats.reader import _dequantize_one  # noqa: E402
from rwkv_quant.formats.schema import QuantizedTensor  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def eq(name, a_np, b_torch, extra=""):
    """numpy-результат против torch-результата, точное равенство."""
    b = b_torch.numpy()
    same = a_np.shape == b.shape and a_np.dtype == b.dtype and bool((a_np == b).all())
    if not same and a_np.shape == b.shape:
        extra = extra or f"dtype {a_np.dtype}/{b.dtype}, расх. {(a_np != b).sum()}"
    check(name, same, extra)


# ---------------- ВЕРБАТИМНЫЕ прежние torch-реализации ----------------

def _ref_pack_int4(codes):
    assert codes.dtype == torch.int8
    rows, cols = codes.shape
    if cols % 2:
        codes = torch.cat([codes, torch.zeros(rows, 1, dtype=torch.int8)], dim=1)
    half = codes.shape[1] // 2
    u = (codes.to(torch.int16) + 8).to(torch.uint8)
    lo, hi = u[:, :half], u[:, half:]
    return lo | (hi << 4)


def _ref_unpack_int4(packed, n_cols):
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    out = torch.cat([lo, hi], dim=1).to(torch.int8)
    return out[:, :n_cols].contiguous()


def _ref_pack6(v):
    x = v.to(torch.int32).reshape(*v.shape[:-1], -1, 4)
    b0 = (x[..., 0] | (x[..., 1] << 6)) & 0xFF
    b1 = ((x[..., 1] >> 2) | (x[..., 2] << 4)) & 0xFF
    b2 = ((x[..., 2] >> 4) | (x[..., 3] << 2)) & 0xFF
    return torch.stack([b0, b1, b2], dim=-1).reshape(*v.shape[:-1], -1).to(torch.uint8)


def _ref_unpack6(b, n):
    x = b.to(torch.int32).reshape(*b.shape[:-1], -1, 3)
    v0 = x[..., 0] & 0x3F
    v1 = ((x[..., 0] >> 6) | (x[..., 1] << 2)) & 0x3F
    v2 = ((x[..., 1] >> 4) | (x[..., 2] << 4)) & 0x3F
    v3 = (x[..., 2] >> 2) & 0x3F
    out = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*b.shape[:-1], -1)
    return out[..., :n].to(torch.uint8)


def _ref_pack_nib_block(q, gs=32):
    OUT, IN = q.shape
    h = gs // 2
    qb = q.view(OUT, IN // gs, gs)
    return (qb[:, :, :h] | (qb[:, :, h:] << 4)).reshape(OUT, IN // 2).contiguous()


def _ref_unpack_nib_block(p, gs=32):
    OUT, HB = p.shape
    h = gs // 2
    pb = p.view(OUT, HB // h, h)
    lo, hi = pb & 0xF, pb >> 4
    return torch.cat([lo, hi], dim=2).reshape(OUT, HB * 2).contiguous()


def _ref_pack_bitplane(bit):
    OUT, IN = bit.shape
    b = bit.to(torch.uint8).view(OUT, IN // 8, 8)
    sh = torch.arange(8, dtype=torch.uint8)
    return (b << sh).sum(dim=2, dtype=torch.int32).to(torch.uint8)


def _ref_unpack_bitplane(p, n_cols):
    OUT = p.shape[0]
    sh = torch.arange(8, dtype=torch.uint8)
    bits = (p.unsqueeze(-1) >> sh) & 1
    return bits.reshape(OUT, -1)[:, :n_cols].contiguous()


def _ref_int8_codes(qt):
    if qt.codes is not None:
        return qt.codes
    return _ref_unpack_int4(qt.codes_packed, qt.shape[1])


def _ref_dequantize_one(qt):
    """Деквант ЦЕЛИКОМ на прежнем torch-коде, включая распаковку. Нужен,
    чтобы доказать: перевод schema-упаковщиков на numpy-обёртки не сдвинул
    выход reader'а ни на бит."""
    if qt.bits >= 16:
        return qt.dense
    if qt.gw_mode == "sb6":
        OUT, IN = qt.shape
        gs, NB = qt.gw_gs, IN // qt.gw_gs
        q = _ref_unpack_nib_block(qt.codes_packed, gs).to(torch.float32)
        if qt.gw_qh is not None:
            q = q + _ref_unpack_bitplane(qt.gw_qh, IN).to(torch.float32) * 16.0
        if qt.gw_qh2 is not None:
            q = q + _ref_unpack_bitplane(qt.gw_qh2, IN).to(torch.float32) * 32.0
        qs = _ref_unpack6(qt.gw_qsqm[..., :6], 8).reshape(OUT, NB).to(torch.float32)
        qm = (_ref_unpack6(qt.gw_qsqm[..., 6:], 8).reshape(OUT, NB).to(torch.int16)
              - 31).to(torch.float32)
        d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)
        dm = qt.gw_dm.float().repeat_interleave(qt.gw_sb, dim=1)
        scale = (qs * d).half().float().clamp_min(1e-8)
        mn = (qm * dm).half().float()
        return (q * scale.repeat_interleave(gs, dim=1)
                + mn.repeat_interleave(gs, dim=1)).to(torch.bfloat16)
    if qt.gw_mode == "asym":
        OUT, IN = qt.shape
        idx = torch.arange(IN) // qt.gw_gs
        return (qt.codes.to(torch.float32) * qt.gw_scale[:, idx]
                + qt.gw_min[:, idx]).to(torch.bfloat16)
    w = (_ref_int8_codes(qt).float() * qt.scale.float()).to(torch.bfloat16)
    if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
        rows, cols = qt.outlier_indices[:, 0].long(), qt.outlier_indices[:, 1].long()
        w[rows, cols] = qt.outlier_values
    return w


# ---------------- torch-free деквант через codec ----------------

def _n(t):
    return None if t is None else t.detach().cpu().contiguous().numpy()


def codec_dequantize_one(qt):
    """Тот же QuantizedTensor, восстановленный БЕЗ torch. Torch тут только
    на входе (распаковать поля дата-класса) и на выходе (bf16, которого у
    numpy нет) -- вся арифметика внутри codec."""
    if qt.bits >= 16:
        return qt.dense
    if qt.gw_mode == "sb6":
        w = codec.dequant_sb6(
            _n(qt.codes_packed), _n(qt.gw_qsqm), _n(qt.gw_d), _n(qt.gw_dm),
            shape=tuple(qt.shape), gs=qt.gw_gs, sb=qt.gw_sb,
            qh=_n(qt.gw_qh), qh2=_n(qt.gw_qh2))
    elif qt.gw_mode == "asym":
        w = codec.dequant_asym(
            _n(qt.codes), _n(qt.gw_scale), _n(qt.gw_min),
            shape=tuple(qt.shape), gs=qt.gw_gs)
    else:
        oi = qt.outlier_indices
        has_out = oi is not None and oi.numel() > 0
        w = codec.dequant_rtn(
            _n(qt.scale), shape=tuple(qt.shape),
            codes=_n(qt.codes), codes_packed=_n(qt.codes_packed),
            outlier_indices=_n(oi) if has_out else None,
            # выбросы хранятся в bf16 -- роундтрип через float32 точен
            outlier_values=(qt.outlier_values.float().numpy()
                            if has_out else None))
    return torch.from_numpy(w).to(torch.bfloat16)


# ---------------- 1. роундтрипы ----------------

def test_roundtrips(rng):
    print("роундтрипы pack -> unpack")

    for cols in (8, 9, 64, 4095):
        c = rng.integers(-8, 8, size=(7, cols), dtype=np.int8)
        back = codec.unpack_int4(codec.pack_int4(c), cols)
        check(f"int4 cols={cols}", bool((back == c).all()))

    for n in (8, 16, 64):
        v = rng.integers(0, 64, size=(5, 3, n), dtype=np.uint8)
        back = codec.unpack6(codec.pack6(v), n)
        check(f"pack6 n={n}", bool((back == v).all()))

    for gs, IN in ((32, 256), (64, 512), (32, 2048)):
        q = rng.integers(0, 16, size=(9, IN), dtype=np.uint8)
        back = codec.unpack_nib_block(codec.pack_nib_block(q, gs), gs)
        check(f"nib_block gs={gs} IN={IN}", bool((back == q).all()))

    for IN in (8, 64, 2048):
        b = rng.integers(0, 2, size=(6, IN), dtype=np.uint8)
        back = codec.unpack_bitplane(codec.pack_bitplane(b), IN)
        check(f"bitplane IN={IN}", bool((back == b).all()))


# ---------------- 2. numpy против прежнего torch ----------------

def test_vs_torch(rng):
    """codec против ЖИВЫХ schema.* -- и заодно живые schema.* против
    замороженных _ref_*, иначе синхронная правка обоих файлов прошла бы
    гейт незамеченной."""
    print("\ncodec (numpy) против schema (torch), schema против замороженной копии")

    def both(name, np_out, live, frozen):
        eq(f"{name} [codec]", np_out, live)
        eq(f"{name} [schema против копии]", live.numpy(), frozen)

    for cols in (8, 9, 64, 1023):
        c = rng.integers(-8, 8, size=(7, cols), dtype=np.int8)
        ct = torch.from_numpy(c)
        both(f"pack_int4 cols={cols}", codec.pack_int4(c),
             schema.pack_int4(ct), _ref_pack_int4(ct))
        p = codec.pack_int4(c)
        pt = torch.from_numpy(p)
        both(f"unpack_int4 cols={cols}", codec.unpack_int4(p, cols),
             schema.unpack_int4(pt, cols), _ref_unpack_int4(pt, cols))

    for shape in ((5, 8), (5, 3, 16), (17, 2, 64)):
        v = rng.integers(0, 64, size=shape, dtype=np.uint8)
        vt = torch.from_numpy(v)
        both(f"pack6 {shape}", codec.pack6(v), schema.pack6(vt), _ref_pack6(vt))
        # распаковка произвольных байтов, не только продуктов pack6
        b = rng.integers(0, 256, size=shape[:-1] + (shape[-1] // 4 * 3,),
                         dtype=np.uint8)
        bt = torch.from_numpy(b)
        n = shape[-1]
        both(f"unpack6 {shape}", codec.unpack6(b, n),
             schema.unpack6(bt, n), _ref_unpack6(bt, n))

    for gs, IN in ((32, 256), (64, 512), (32, 2048)):
        q = rng.integers(0, 16, size=(9, IN), dtype=np.uint8)
        qt = torch.from_numpy(q)
        both(f"pack_nib_block gs={gs} IN={IN}", codec.pack_nib_block(q, gs),
             schema.pack_nib_block(qt, gs), _ref_pack_nib_block(qt, gs))
        p = rng.integers(0, 256, size=(9, IN // 2), dtype=np.uint8)
        pt = torch.from_numpy(p)
        both(f"unpack_nib_block gs={gs} IN={IN}", codec.unpack_nib_block(p, gs),
             schema.unpack_nib_block(pt, gs), _ref_unpack_nib_block(pt, gs))

    for IN in (8, 64, 2048):
        b = rng.integers(0, 2, size=(6, IN), dtype=np.uint8)
        bt = torch.from_numpy(b)
        both(f"pack_bitplane IN={IN}", codec.pack_bitplane(b),
             schema.pack_bitplane(bt), _ref_pack_bitplane(bt))
        p = rng.integers(0, 256, size=(6, IN // 8), dtype=np.uint8)
        pt = torch.from_numpy(p)
        both(f"unpack_bitplane IN={IN}", codec.unpack_bitplane(p, IN),
             schema.unpack_bitplane(pt, IN), _ref_unpack_bitplane(pt, IN))


# ---------------- 3. деквант синтетических тензоров ----------------

def _mk_sb6(rng, OUT, IN, gs, sb, xbits):
    NB, NSB = IN // gs, IN // (gs * sb)
    t = torch.from_numpy
    qt = QuantizedTensor(
        key="synth", group="proj", bits=4 + xbits, shape=(OUT, IN),
        gw_mode="sb6", gw_gs=gs, gw_sb=sb,
        codes_packed=t(rng.integers(0, 256, (OUT, IN // 2), dtype=np.uint8)),
        gw_qsqm=t(rng.integers(0, 256, (OUT, NSB, 12), dtype=np.uint8)),
        # масштабы в правдоподобном диапазоне: околонулевые d вывели бы
        # clamp_min на первый план и спрятали расхождения в мантиссе
        gw_d=t(rng.uniform(1e-4, 1e-2, (OUT, NSB)).astype(np.float16)),
        gw_dm=t(rng.uniform(-1e-2, 1e-2, (OUT, NSB)).astype(np.float16)),
    )
    if xbits >= 1:
        qt.gw_qh = t(rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8))
    if xbits >= 2:
        qt.gw_qh2 = t(rng.integers(0, 256, (OUT, IN // 8), dtype=np.uint8))
    return qt


def test_dequant(rng):
    print("\nдеквант: codec (numpy) против reader (torch)")
    t = torch.from_numpy
    cases = []

    for gs, sb, xbits in ((32, 8, 0), (32, 8, 1), (32, 8, 2), (64, 8, 2)):
        cases.append((f"sb6 gs={gs} xbits={xbits}",
                      _mk_sb6(rng, 48, gs * sb * 3, gs, sb, xbits)))

    # asym: с точным NB и с паддингом хвостовых блоков (NBpad)
    for pad in (0, 3):
        OUT, IN, gs = 32, 512, 64
        NB = IN // gs
        cases.append((f"asym pad={pad}", QuantizedTensor(
            key="synth", group="w_lora", bits=6, shape=(OUT, IN),
            gw_mode="asym", gw_gs=gs,
            codes=t(rng.integers(0, 64, (OUT, IN), dtype=np.uint8)),
            gw_scale=t(rng.uniform(1e-4, 1e-2, (OUT, NB + pad)).astype(np.float32)),
            gw_min=t(rng.uniform(-1e-2, 1e-2, (OUT, NB + pad)).astype(np.float32)),
        )))

    # rtn int8 per-row, в том числе 3-D (writer квантует всё с dim >= 2)
    for shape in ((64, 512), (1, 1, 256)):
        cases.append((f"rtn int8 {shape}", QuantizedTensor(
            key="synth", group="cmix", bits=8, shape=shape,
            codes=t(rng.integers(-128, 128, shape, dtype=np.int8)),
            scale=t(rng.uniform(1e-4, 1e-2, (shape[0],) + (1,) * (len(shape) - 1))
                    .astype(np.float16)),
        )))

    # rtn упакованный (bits <= 4), чётное и нечётное число колонок
    for IN in (512, 511):
        codes = rng.integers(-8, 8, (64, IN), dtype=np.int8)
        cases.append((f"rtn packed IN={IN}", QuantizedTensor(
            key="synth", group="cmix", bits=4, shape=(64, IN),
            codes_packed=t(codec.pack_int4(codes)),
            scale=t(rng.uniform(1e-4, 1e-2, (64, 1)).astype(np.float16)),
        )))

    # rtn + SpQR-надстройка
    n_out = 40
    oi = np.stack([rng.integers(0, 64, n_out), rng.integers(0, 512, n_out)],
                  axis=1).astype(np.int32)
    cases.append(("rtn + outliers", QuantizedTensor(
        key="synth", group="cmix", bits=8, shape=(64, 512),
        codes=t(rng.integers(-128, 128, (64, 512), dtype=np.int8)),
        scale=t(rng.uniform(1e-4, 1e-2, (64, 1)).astype(np.float16)),
        outlier_indices=t(oi),
        outlier_values=t(rng.uniform(-1, 1, n_out).astype(np.float32)).bfloat16(),
    )))

    cases.append(("dense", QuantizedTensor(
        key="synth", group="other", bits=16, shape=(1, 1, 256),
        dense=t(rng.uniform(-1, 1, (1, 1, 256)).astype(np.float32)).bfloat16(),
    )))

    for name, qt in cases:
        cur = _dequantize_one(qt)                 # reader сегодня
        old = _ref_dequantize_one(qt)             # reader до рефакторинга
        new = codec_dequantize_one(qt)            # torch-free нормативный
        if not (cur.shape == old.shape == new.shape
                and cur.dtype == old.dtype == new.dtype):
            check(name, False, f"формы/типы {tuple(cur.shape)}/{cur.dtype}, "
                               f"{tuple(old.shape)}/{old.dtype}, "
                               f"{tuple(new.shape)}/{new.dtype}")
            continue
        n_old = int((cur != old).sum())
        n_new = int((cur != new).sum())
        check(name, n_old == 0 and n_new == 0,
              "" if n_old == 0 and n_new == 0
              else f"против прежнего torch {n_old}, против codec {n_new}")


def test_real(path):
    """Те же три реализации на настоящем чекпоинте: синтетика не покрывает
    реальное распределение масштабов и вырожденные блоки (нулевой d,
    упершийся clamp_min, все коды одинаковы)."""
    from rwkv_quant.formats.reader import load_raw
    print(f"\nдеквант на реальном чекпоинте: {path}")
    ckpt = load_raw(path)
    seen, bad, n_el = {}, 0, 0
    for key, qt in ckpt.tensors.items():
        kind = ("dense" if qt.bits >= 16 else qt.gw_mode or
                ("rtn_packed" if qt.codes_packed is not None else "rtn"))
        cur = _dequantize_one(qt)
        old, new = _ref_dequantize_one(qt), codec_dequantize_one(qt)
        n = (int((cur != old).sum()) + int((cur != new).sum())
             if cur.shape == old.shape == new.shape else -1)
        n_el += cur.numel()
        s = seen.setdefault(kind, [0, 0])
        s[0] += 1
        if n != 0:
            s[1] += 1
            bad += 1
            if s[1] <= 3:
                print(f"  !! {key} ({kind}): расхождений {n}")
        del cur, old, new
    for kind, (total, nb) in sorted(seen.items()):
        check(f"real {kind}", nb == 0, f"{total} тензоров"
              if nb == 0 else f"{nb} из {total} разошлись")
    print(f"  всего {sum(v[0] for v in seen.values())} тензоров, "
          f"{n_el / 1e6:.1f}M элементов, расхождений {bad}")


def main():
    rng = np.random.default_rng(20260804)
    test_roundtrips(rng)
    test_vs_torch(rng)
    test_dequant(rng)
    for path in sys.argv[1:]:
        test_real(path)
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else f"ПРОВАЛЕН: {', '.join(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
