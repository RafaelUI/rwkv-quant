"""
Загрузка .rwkvq. Два режима:
  - load_raw(path)         -> QuantizedCheckpoint как есть (для backends/,
                               которые будут делать реальный low-bit инференс
                               напрямую на codes/scale, без деквантования)
  - load_dequantized(path) -> обычный bf16 state_dict, готовый для
                               RWKV7Ref(...) -- нужен для валидации/сравнения
                               ppl квантованной модели с оригиналом.

TORCH-FREE ДВОЙНИК (04.08). Нормативная реализация того же декванта без
torch -- codec.dequant_sb6 / dequant_asym / dequant_rtn; именно с неё
портируются rwkv-metal и SwiftRWKV. Здешний код остался torch'евым не по
инерции, а по замеру: tests/bench_codec_dequant_ab.py на 344M элементов
даёт torch 289 мс против numpy 760 мс (2.63x) -- torch распараллеливает
поэлементные операции по ядрам, numpy нет.

Что обе реализации совпадают бит-в-бит, доказывает
tests/test_codec_parity.py: 2.9 G элементов чекпоинта 2.9B, все четыре
раскладки, ноль расхождений. При правке раскладки править ОБЕ и гонять
гейт.
"""
import json

import torch

from . import codec
from .schema import (QuantizedCheckpoint, QuantizedTensor,  # noqa: F401
                     int8_codes, unpack6, unpack_nib_block, unpack_bitplane)


def load_raw(path: str) -> QuantizedCheckpoint:
    """.rwkvq -> QuantizedCheckpoint. Читает ОБА контейнера.

    Различаются по первым байтам: torch.save пишет zip (PK\\x03\\x04),
    safetensors -- little-endian u64 с длиной JSON-хедера. Прежние файлы
    поддерживаются один релиз; новые пишутся только в safetensors
    (writer.save_rwkvq -- там же и мотивация).
    """
    with open(path, "rb") as f:
        head = f.read(4)
    if head == codec.MAGIC_ZIP:
        # pickle: исполняет код при загрузке и требует установленного
        # rwkv_quant той же версии -- ровно то, от чего уходим
        return torch.load(path, map_location="cpu", weights_only=False)
    return _load_safetensors(path)


def _load_safetensors(path: str) -> QuantizedCheckpoint:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if "rwkvq" not in meta:
            raise ValueError(f"{path}: safetensors без манифеста rwkvq")
        m = json.loads(meta["rwkvq"])
        if m.get("format") != codec.FORMAT:
            raise ValueError(f"{path}: формат {m.get('format')!r}, "
                             f"ожидался {codec.FORMAT!r}")
        if m.get("format_version", 0) > codec.FORMAT_VERSION:
            raise ValueError(
                f"{path}: format_version {m['format_version']} новее, чем "
                f"понимает этот rwkv_quant ({codec.FORMAT_VERSION})")

        tensors = {}
        for key, t in m["tensors"].items():
            kind = t["kind"]
            qt = QuantizedTensor(
                key=key, group=t["group"], bits=t["bits"],
                shape=tuple(t["shape"]),
                gw_mode=kind if kind in ("sb6", "asym") else "",
                gw_gs=t["gw_gs"], gw_sb=t["gw_sb"])
            for field in t["fields"]:
                setattr(qt, field, f.get_tensor(f"{key}::{field}"))
            tensors[key] = qt

    return QuantizedCheckpoint(
        naming=m["naming"], n_layer=m["n_layer"], n_embd=m["n_embd"],
        head_size=m["head_size"], vocab_size=m["vocab_size"],
        tensors=tensors, config_repr=m.get("config_repr", ""),
        # v1 этих полей не имеет -- отсутствие тут норма, а не ошибка
        config=config_from_json(m.get("config")),
        tokenizer=m.get("tokenizer"))


def config_from_json(d):
    """Обратно к QuantConfig из структуры манифеста (см.
    writer.config_to_json). None -> None: у файлов v1 конфиг был записан
    только лосси-строкой repr(), собрать из неё нечего."""
    if d is None:
        return None
    from ..calibration.group_config import QuantConfig
    cfg = QuantConfig(
        clip_percentiles=d.get("clip_percentiles"),
        outlier_fracs=d.get("outlier_fracs"),
        bits_overrides=d.get("bits_overrides"),
        group_scale=d.get("group_scale"),
        group_scale_mode=d.get("group_scale_mode"),
        act_stats_path=d.get("act_stats_path"),
        **d.get("bits", {}))
    return cfg


def _dequantize_one(qt) -> torch.Tensor:
    if qt.bits >= 16:
        return qt.dense
    if qt.gw_mode == "sb6":
        return _dequantize_gw_sb6(qt)
    if qt.gw_mode == "asym":
        return _dequantize_gw_asym(qt)
    w = (int8_codes(qt).float() * qt.scale.float()).to(torch.bfloat16)
    if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
        rows, cols = qt.outlier_indices[:, 0].long(), qt.outlier_indices[:, 1].long()
        w[rows, cols] = qt.outlier_values
    return w


def load_dequantized(path: str) -> dict:
    ckpt = load_raw(path)
    return {key: _dequantize_one(qt) for key, qt in ckpt.tensors.items()}


def _dequantize_gw_sb6(qt) -> torch.Tensor:
    """Формат v2: восстановление в точности по формуле кернеля --
    s = half(qs * float(d_half)), m = half(qm * float(dm_half)),
    w = q * s + m; clamp scale как в writer (см. NaN-примечание там)."""
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
    d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)    # [OUT, NB]
    dm = qt.gw_dm.float().repeat_interleave(qt.gw_sb, dim=1)
    scale = (qs * d).half().float().clamp_min(1e-8)
    mn = (qm * dm).half().float()
    scale_c = scale.repeat_interleave(gs, dim=1)
    mn_c = mn.repeat_interleave(gs, dim=1)
    return (q * scale_c + mn_c).to(torch.bfloat16)


def _dequantize_gw_asym(qt) -> torch.Tensor:
    OUT, IN = qt.shape
    gs = qt.gw_gs
    q = qt.codes.to(torch.float32)          # uint8-контейнер, unsigned коды
    idx = torch.arange(IN) // gs
    scale_c = qt.gw_scale[:, idx]
    mn_c = qt.gw_min[:, idx]
    return (q * scale_c + mn_c).to(torch.bfloat16)
