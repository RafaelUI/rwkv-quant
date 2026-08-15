"""
Экспорт .rwkvq в torch-free MLX-сайдкар: safetensors + JSON-манифест.

Зачем: .rwkvq -- это torch.save питоновского объекта, то есть pickle. Его
нельзя прочитать ни без torch, ни без установленного пакета rwkv_quant
(pickle пишет полные имена классов). Потребители формата -- rwkv-metal и
SwiftRWKV -- принципиально torch-free, поэтому нужен мост.

ПОЛНЫЙ ЭКСПОРТ (04.08). Раньше сюда попадали ТОЛЬКО тензоры с
gw_mode="sb6" -- 194 из 1062 на 2.9B. Всё остальное (asym-lora, per-row
int8, одномерные нормы и token-shift миксы) приходилось брать из
исходного bf16-чекпоинта, и приложение вынуждено было качать его целиком:
5895 МБ bf16 + 1796 МБ сайдкара = 7691 МБ там, где хватает 1898.
Разница -- 102 МБ на 2.9B, то есть +5.7% к сайдкару, чтобы 5.9 ГБ
исчезли. Теперь экспортируется КАЖДЫЙ тензор чекпоинта.

Раскладки (поле "kind" в манифесте определяет, какие буферы искать):

  "sb6"    -- K3-интерлив: qblk / qsqm / ddm. Та же раскладка, что у
              backends/metal/quant_linear_gw.py::GwQuantLinear, уже
              провалидирована бит-в-бит с диск-форматом. Стоит +0.125
              бит/вес против канонической (qsqm как uchar2 вместо 12
              бит) -- на 2.9B это +45 МБ, платим за то, что грузится
              mmap'ом как есть, без репака.
  "sym"    -- Q6_K: qblk (коды блоками по 16) + qs (int8 на блок) +
              d (fp16 на суперблок). При bits=8 в qblk знаковые байты,
              при bits=6 -- ниббл со сдвигом +32 и две битплоскости.
  "asym"   -- gw-asym (LoRA @6, gw64): codes (uint8-контейнер) +
              scale / min (fp32 на блок).
  "rtn"    -- per-row RTN: codes (int8) ЛИБО codes_packed (uint8-нибблы
              при bits<=4) + scale (fp16 [rows,1]); опционально
              outlier_idx (int32 [n,2]) + outlier_val (bf16) для SpQR.
  "dense"  -- bf16 as-is (bits>=16): одномерные параметры, нормы,
              bias-термы LoRA-веток, всё что writer не квантует.

bf16 идёт в safetensors как есть: numpy bfloat16 не умеет, поэтому
конверсия через float32 (точная -- bf16 является подмножеством fp32).

Одноразовый шаг в venv rwkv-quant (там есть torch):
  python -m rwkv_quant.formats.export_mlx model.rwkvq out/model_mlx

Выход: out/model_mlx.safetensors + out/model_mlx.json
"""
import json
import os
import sys

import mlx.core as mx
import torch

from .reader import load_raw
from ..backends.metal.quant_linear_gw import GwQuantLinear
from ..backends.metal.quant_linear_sym import SymQuantLinear

_DTYPE = {
    torch.uint8: mx.uint8, torch.int8: mx.int8, torch.int32: mx.int32,
    torch.float16: mx.float16, torch.float32: mx.float32,
}


def _to_mx(t: torch.Tensor) -> mx.array:
    """torch -> mlx без потерь. bf16 -- через float32: numpy не знает
    bfloat16, а bf16 точно представим в fp32, так что роундтрип точен."""
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy()).astype(mx.bfloat16)
    dt = _DTYPE.get(t.dtype)
    if dt is None:
        raise TypeError(f"неожиданный dtype {t.dtype}")
    return mx.array(t.numpy()).astype(dt)


def _export_one(key, qt, tensors):
    """Один QuantizedTensor -> буферы в tensors + метаданные."""
    meta = {"shape": list(qt.shape), "bits": int(qt.bits),
            "group": qt.group or "other"}

    if qt.bits >= 16:
        tensors[f"{key}::dense"] = _to_mx(qt.dense)
        meta["kind"] = "dense"
        return meta

    if qt.gw_mode == "sb6":
        gw = GwQuantLinear(qt)
        assert gw._k3, f"{key}: K3-интерлив не построился (OUT%16 != 0?)"
        tensors[f"{key}::qblk"] = gw.qblk
        tensors[f"{key}::qsqm"] = gw.qsqm
        tensors[f"{key}::ddm"] = gw.ddm
        meta.update(kind="sb6", xbits=int(gw.xbits),
                    gw_gs=int(qt.gw_gs), gw_sb=int(qt.gw_sb))
        return meta

    if qt.gw_mode == "sym":
        # Q6_K: интерлив загрузчика (qblk) + int8-масштабы блоков + fp16 d
        # суперблока. Без этой ветки sym-тензор проваливался бы в rtn ниже
        # и уехал бы в сайдкар с чужим kind и без scale -- то есть молча
        # неверным, а не сломанным (закон 15).
        lin = SymQuantLinear(qt)
        tensors[f"{key}::qblk"] = lin.qblk
        tensors[f"{key}::qs"] = lin.qs
        tensors[f"{key}::d"] = lin.d
        meta.update(kind="sym", gw_gs=int(qt.gw_gs), gw_sb=int(qt.gw_sb))
        return meta

    if qt.gw_mode == "asym":
        tensors[f"{key}::codes"] = _to_mx(qt.codes)
        tensors[f"{key}::gw_scale"] = _to_mx(qt.gw_scale)
        tensors[f"{key}::gw_min"] = _to_mx(qt.gw_min)
        meta.update(kind="asym", gw_gs=int(qt.gw_gs))
        return meta

    # per-row RTN: коды int8 либо упакованные нибблы, плюс scale и
    # опциональная разреженная SpQR-надстройка
    meta["kind"] = "rtn"
    if qt.codes_packed is not None:
        tensors[f"{key}::codes_packed"] = _to_mx(qt.codes_packed)
        meta["packed"] = True
    else:
        tensors[f"{key}::codes"] = _to_mx(qt.codes)
        meta["packed"] = False
    tensors[f"{key}::scale"] = _to_mx(qt.scale)
    if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0:
        tensors[f"{key}::outlier_idx"] = _to_mx(qt.outlier_indices)
        tensors[f"{key}::outlier_val"] = _to_mx(qt.outlier_values)
        meta["outliers"] = int(qt.outlier_indices.shape[0])
    return meta


def export(rwkvq_path: str, out_path: str, tokenizer: str = None):
    ckpt = load_raw(rwkvq_path)
    tensors = {}
    manifest = {
        "format": "rwkvq_mlx",
        "format_version": 2,          # 1 -- только sb6, до 04.08
        "naming": ckpt.naming, "n_layer": ckpt.n_layer, "n_embd": ckpt.n_embd,
        "head_size": ckpt.head_size, "vocab_size": ckpt.vocab_size,
        "config_repr": ckpt.config_repr,
        "tokenizer": tokenizer,
        "tensors": {},
    }
    counts = {}
    for key, qt in ckpt.tensors.items():
        meta = _export_one(key, qt, tensors)
        manifest["tensors"][key] = meta
        counts[meta["kind"]] = counts.get(meta["kind"], 0) + 1

    mx.save_safetensors(out_path, tensors)
    with open(out_path + ".json", "w") as f:
        json.dump(manifest, f, indent=2)

    size = os.path.getsize(out_path if out_path.endswith(".safetensors")
                           else out_path + ".safetensors") / 1e6
    print(f"экспортировано {len(manifest['tensors'])} тензоров "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}) "
          f"-> {size:.1f} МБ + манифест")
    return manifest


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2],
           tokenizer=sys.argv[3] if len(sys.argv) > 3 else None)
