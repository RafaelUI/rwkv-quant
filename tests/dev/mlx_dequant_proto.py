"""
Прототип MLX-порта деквантования .rwkvq (gw_mode sb6/asym) для QLoRA-базы.
Сверяется бит-в-бит (точнее, численно, т.к. bf16 округления могут отличаться
по порядку операций) с PyTorch-референсом rwkv_quant.formats.reader.

Не финальный модуль -- песочница для проверки перед переносом в пакет.
"""
import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")

import numpy as np
import torch
import mlx.core as mx

from rwkv_quant.formats import reader
from rwkv_quant.formats.schema import unpack6 as unpack6_torch


def t2mx(t: torch.Tensor) -> mx.array:
    """torch tensor (cpu) -> mx.array, без деквантования, сохраняя dtype по смыслу."""
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.uint16).numpy()).view(mx.bfloat16)
    if t.dtype == torch.float16:
        return mx.array(t.numpy().astype(np.float16))
    if t.dtype == torch.int8:
        return mx.array(t.numpy().astype(np.int8))
    if t.dtype == torch.uint8:
        return mx.array(t.numpy().astype(np.uint8))
    if t.dtype == torch.int32:
        return mx.array(t.numpy().astype(np.int32))
    if t.dtype == torch.float32:
        return mx.array(t.numpy().astype(np.float32))
    raise TypeError(f"unhandled dtype {t.dtype}")


def unpack_nib_block_mlx(p: mx.array, gs: int) -> mx.array:
    OUT, HB = p.shape
    h = gs // 2
    pb = p.reshape(OUT, HB // h, h)
    lo = pb & 0xF
    hi = pb >> 4
    return mx.concatenate([lo, hi], axis=2).reshape(OUT, HB * 2)


def unpack_bitplane_mlx(p: mx.array, n_cols: int) -> mx.array:
    OUT = p.shape[0]
    p32 = p.astype(mx.uint32)
    sh = mx.arange(8, dtype=mx.uint32)
    bits = (p32[:, :, None] >> sh) & 1
    return bits.reshape(OUT, -1)[:, :n_cols].astype(mx.uint8)


def unpack6_mlx(b: mx.array, n: int) -> mx.array:
    # b: [..., K*3] uint8 -> [..., K*4] uint8 (0..63), обрезано до n
    lead = b.shape[:-1]
    K = b.shape[-1] // 3
    x = b.astype(mx.int32).reshape(*lead, K, 3)
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    v0 = x0 & 0x3F
    v1 = ((x0 >> 6) | (x1 << 2)) & 0x3F
    v2 = ((x1 >> 4) | (x2 << 4)) & 0x3F
    v3 = (x2 >> 2) & 0x3F
    out = mx.stack([v0, v1, v2, v3], axis=-1).reshape(*lead, K * 4)
    return out[..., :n].astype(mx.uint8)


def dequantize_gw_sb6_mlx(qt) -> mx.array:
    OUT, IN = qt.shape
    gs, NB = qt.gw_gs, IN // qt.gw_gs

    codes_packed = t2mx(qt.codes_packed)
    q = unpack_nib_block_mlx(codes_packed, gs).astype(mx.float32)

    if qt.gw_qh is not None:
        qh = t2mx(qt.gw_qh)
        q = q + unpack_bitplane_mlx(qh, IN).astype(mx.float32) * 16.0
    if qt.gw_qh2 is not None:
        qh2 = t2mx(qt.gw_qh2)
        q = q + unpack_bitplane_mlx(qh2, IN).astype(mx.float32) * 32.0

    qsqm = t2mx(qt.gw_qsqm)  # [OUT, NSB, 12] uint8
    qs = unpack6_mlx(qsqm[..., :6], 8).reshape(OUT, NB).astype(mx.float32)
    qm = unpack6_mlx(qsqm[..., 6:], 8).reshape(OUT, NB).astype(mx.int32).astype(mx.float32) - 31.0

    d = t2mx(qt.gw_d).astype(mx.float32)   # [OUT, NSB] fp16 -> f32
    dm = t2mx(qt.gw_dm).astype(mx.float32)
    d_c = mx.repeat(d, qt.gw_sb, axis=1)     # [OUT, NB]
    dm_c = mx.repeat(dm, qt.gw_sb, axis=1)

    # half-round-trip как в референсе: (qs*d).half().float()
    scale = (qs * d_c).astype(mx.float16).astype(mx.float32)
    scale = mx.maximum(scale, 1e-8)
    mn = (qm * dm_c).astype(mx.float16).astype(mx.float32)

    scale_c = mx.repeat(scale, gs, axis=1)
    mn_c = mx.repeat(mn, gs, axis=1)
    w = q * scale_c + mn_c
    return w.astype(mx.bfloat16)


def dequantize_gw_asym_mlx(qt) -> mx.array:
    OUT, IN = qt.shape
    gs = qt.gw_gs
    q = t2mx(qt.codes).astype(mx.float32)  # uint8 container per docstring (unsigned)
    idx = mx.arange(IN) // gs
    scale = t2mx(qt.gw_scale).astype(mx.float32)  # [OUT, NB]
    mn = t2mx(qt.gw_min).astype(mx.float32)
    scale_c = scale[:, idx]
    mn_c = mn[:, idx]
    w = q * scale_c + mn_c
    return w.astype(mx.bfloat16)


def compare(name, w_ref: torch.Tensor, w_mlx: mx.array):
    ref = w_ref.float().numpy()
    got = np.array(w_mlx.astype(mx.float32))
    diff = np.abs(ref - got)
    rel = diff / (np.abs(ref) + 1e-6)
    n_exact_bf16 = int((w_ref.view(torch.uint16).numpy() ==
                         mx.array(w_mlx).astype(mx.bfloat16).view(mx.uint16).__array__()).sum()) \
        if False else -1
    print(f"[{name}] shape={tuple(w_ref.shape)} max_abs_diff={diff.max():.6g} "
          f"mean_abs_diff={diff.mean():.6g} max_rel={rel.max():.4g} "
          f"n_mismatch(bf16 bits)={(w_ref.view(torch.int16).numpy() != np.array(w_mlx.view(mx.uint16)).astype(np.int16)).sum()} / {w_ref.numel()}")


def main():
    path = "/tmp/reduction_v2.rwkvq"
    print(f"loading {path} ...")
    t0 = time.time()
    ckpt = reader.load_raw(path)
    print(f"loaded in {time.time()-t0:.1f}s, {len(ckpt.tensors)} tensors")

    # найдём по одному представителю sb6 и asym
    sb6_key = None
    asym_key = None
    for k, qt in ckpt.tensors.items():
        if qt.gw_mode == "sb6" and sb6_key is None:
            sb6_key = k
        if qt.gw_mode == "asym" and asym_key is None:
            asym_key = k
        if sb6_key and asym_key:
            break

    print("sb6 sample:", sb6_key)
    print("asym sample:", asym_key)

    if sb6_key:
        qt = ckpt.tensors[sb6_key]
        w_ref = reader._dequantize_gw_sb6(qt)
        w_mlx = dequantize_gw_sb6_mlx(qt)
        compare(f"sb6:{sb6_key}", w_ref, w_mlx)

    if asym_key:
        qt = ckpt.tensors[asym_key]
        w_ref = reader._dequantize_gw_asym(qt)
        w_mlx = dequantize_gw_asym_mlx(qt)
        compare(f"asym:{asym_key}", w_ref, w_mlx)


if __name__ == "__main__":
    main()
