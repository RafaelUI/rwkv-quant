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
import os

import torch

from . import codec
from .schema import (QuantizedCheckpoint, QuantizedTensor,  # noqa: F401
                     int8_codes, unpack6, unpack_nib_block, unpack_bitplane)

# Полоса строк для декванта, в МЕГАБАЙТАХ одной fp32-копии полосы.
# Деквант держит несколько полноразмерных fp32-копий тензора разом (коды,
# битплоскости, развёрнутые scale/min, произведение), поэтому emb 1.5B
# [65536, 2048] стоит порядка 2.5 ГБ транзиента ради 268 МБ результата.
# Это ровно та же статья, что закон 20 закрыл в КВАНТОВАТЕЛЕ, только на
# обратном пути, и лечится тем же: строка независима, значит полосу можно
# считать отдельно и результат бит-в-бит тот же (гейт
# tests/test_dequant_band_parity.py).
DEQUANT_CHUNK_MB = int(os.environ.get("RWKVQ_DEQUANT_CHUNK_MB", "64"))


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
                gw_mode=kind if kind in ("sb6", "sym", "asym") else "",
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
    if qt.gw_mode == "sym":
        return _dequantize_gw_sym(qt)
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


# --- деквант полосами строк -------------------------------------------------

# Буферы всех квантованных раскладок имеют ПЕРВОЙ осью выходной канал.
# Здесь перечислены те, что режутся по строкам вместе с тензором; всё
# остальное (скалярные поля манифеста) переносится как есть.
_ROW_FIELDS = ("codes", "codes_packed", "scale", "gw_qh", "gw_qh2",
               "gw_qsqm", "gw_qs", "gw_d", "gw_dm", "gw_scale", "gw_min")


class _RowBand:
    """Вид на полосу строк [a, b) квантованного тензора.

    Вся математика декванта построчная вдоль ВЫХОДНОЙ оси и поблочная
    вдоль ВХОДНОЙ, поэтому деквант полосы бит-в-бит равен соответствующей
    полосе декванта целого. Аргумент тот же, что у нарезки квантователя
    (закон 20), и, как там, он не принимается на слово: гейт
    tests/test_dequant_band_parity.py требует РАВЕНСТВА на реальных формах.
    """

    def __init__(self, qt, a, b):
        self.key = getattr(qt, "key", "")
        self.group = getattr(qt, "group", "")
        self.bits = qt.bits
        self.gw_mode = getattr(qt, "gw_mode", "")
        self.gw_gs = getattr(qt, "gw_gs", 0)
        self.gw_sb = getattr(qt, "gw_sb", 0)
        self.shape = (b - a, qt.shape[1])
        # полоса берётся только у тензоров БЕЗ выбросов (см. can_band):
        # outlier_indices пришлось бы фильтровать и пересчитывать
        self.outlier_indices = None
        self.outlier_values = None
        for f in _ROW_FIELDS:
            v = getattr(qt, f, None)
            setattr(self, f, None if v is None else v[a:b])


def can_band(qt) -> bool:
    """Режется ли тензор полосами. Требуется ровно двумерная форма,
    реальное квантование и отсутствие выбросов: у per-row RTN с
    `outlier_indices` строки перестают быть независимыми по индексации, а
    выигрыш там нулевой -- такие тензоры мелкие и статьи в памяти не
    делают."""
    if qt.bits >= 16 or len(getattr(qt, "shape", ())) != 2:
        return False
    oi = getattr(qt, "outlier_indices", None)
    return oi is None or oi.numel() == 0


def dequantize_banded(qt, dtype=torch.bfloat16, chunk_mb=None) -> torch.Tensor:
    """То же, что `_dequantize_one(qt).to(dtype)`, но результат собирается
    полосами строк сразу в `dtype`, поэтому пик равен результату плюс одна
    полоса, а не нескольким полноразмерным fp32-копиям.

    Каст внутрь НЕ переносится: полоса считается ровно тем же кодом и в тех
    же fp32/bf16, что и целый тензор, и только потом округляется. Иначе это
    была бы другая схема, а не оптимизация (закон 15)."""
    if not can_band(qt):
        return _dequantize_one(qt).to(dtype)
    OUT, IN = qt.shape
    rows = max(1, int((chunk_mb or DEQUANT_CHUNK_MB) * (1 << 20)) // (IN * 4))
    if rows >= OUT:
        return _dequantize_one(qt).to(dtype)
    out = torch.empty((OUT, IN), dtype=dtype)
    for a in range(0, OUT, rows):
        b = min(a + rows, OUT)
        out[a:b] = _dequantize_one(_RowBand(qt, a, b)).to(dtype)
    return out


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
    if NB * gs == IN:
        # scale/min РАЗВЁРНУТЫМИ не материализуются: broadcast по последней
        # оси даёт те же числа, а держал он две полноразмерные fp32-копии.
        # in-place по q законен -- он наш, выше он создан .to(float32).
        q = q.view(OUT, NB, gs)
        q.mul_(scale[..., None]).add_(mn[..., None])
        return q.view(OUT, IN).to(torch.bfloat16)
    scale_c = scale.repeat_interleave(gs, dim=1)              # ragged: NB*gs > IN
    mn_c = mn.repeat_interleave(gs, dim=1)
    return (q * scale_c + mn_c).to(torch.bfloat16)


def _dequantize_gw_sym(qt) -> torch.Tensor:
    """Q6_K-раскладка: s = half(qs * d), w = q * s. Min нет, поэтому нет
    ни clamp_min для scale, ни второй пары квальных скаляров -- у
    вырожденного блока и scale, и коды нулевые (см. codec.dequant_sym).
    Битность различается ПО БУФЕРАМ: codes есть только при восьми битах."""
    OUT, IN = qt.shape
    gs, NB = qt.gw_gs, IN // qt.gw_gs
    if qt.codes is not None:
        q = qt.codes.to(torch.float32)                 # знаковые байты
    else:
        q = unpack_nib_block(qt.codes_packed, gs).to(torch.int16)
        if qt.gw_qh is not None:
            q = q + unpack_bitplane(qt.gw_qh, IN).to(torch.int16) * 16
        if qt.gw_qh2 is not None:
            q = q + unpack_bitplane(qt.gw_qh2, IN).to(torch.int16) * 32
        q = (q - 32).to(torch.float32)                 # снятие сдвига
    d = qt.gw_d.float().repeat_interleave(qt.gw_sb, dim=1)     # [OUT, NB]
    scale = (qt.gw_qs.float() * d).half().float()
    if NB * gs == IN:
        q = q.view(OUT, NB, gs)
        q.mul_(scale[..., None])                               # см. sb6
        return q.view(OUT, IN).to(torch.bfloat16)
    return (q * scale.repeat_interleave(gs, dim=1)).to(torch.bfloat16)


def _dequantize_gw_asym(qt) -> torch.Tensor:
    OUT, IN = qt.shape
    gs = qt.gw_gs
    q = qt.codes.to(torch.float32)          # uint8-контейнер, unsigned коды
    if IN % gs == 0:
        NB = IN // gs
        q = q.view(OUT, NB, gs)
        q.mul_(qt.gw_scale[..., None]).add_(qt.gw_min[..., None])
        return q.view(OUT, IN).to(torch.bfloat16)
    # ragged (blocks.N.att.w1 [2048, 96] при gs=64 -- блоков ceil, не floor):
    # broadcast тут не выражается, идём прежним путём с развёрнутой индексацией
    idx = torch.arange(IN) // gs
    scale_c = qt.gw_scale[:, idx]
    mn_c = qt.gw_min[:, idx]
    return (q * scale_c + mn_c).to(torch.bfloat16)
