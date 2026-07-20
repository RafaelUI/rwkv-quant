"""
Экспорт .rwkvq (gw_mode="sb6" тензоры) в torch-free MLX-сайдкар для
QLoRA-тренировки в rwkv-metal (та среда не тащит torch).

Одноразовый шаг конвертации (в venv rwkv-quant, где есть torch):
  python -m rwkv_quant.formats.export_mlx /tmp/reduction_v2.rwkvq /tmp/reduction_v2.rwkvq_mlx

Выход: safetensors-файл с K3-интерлив-буферами (qblk/qsqm/ddm -- та же
раскладка, что использует backends/metal/quant_linear_gw.py::GwQuantLinear,
уже провалидирована бит-в-бит с диск-форматом) на КАЖДЫЙ sb6-тензор,
plus JSON-манифест с метаданными (shape, gw_gs, gw_sb, bits, xbits) --
безопасно грузится в rwkv-metal через mx.load, без torch.

НЕ включает asym-тензоры (w/a/v_lora) -- по конвенции rwkv-metal
(BIG_QUANT_TARGETS/TMIX_TARGETS) низкоранговые lora-матрицы не квантуются
для QLoRA-базы, остаются fp.
"""
import sys, json
import mlx.core as mx

from .reader import load_raw
from ..backends.metal.quant_linear_gw import GwQuantLinear


def export(rwkvq_path: str, out_path: str):
    ckpt = load_raw(rwkvq_path)
    tensors = {}
    manifest = {
        "naming": ckpt.naming, "n_layer": ckpt.n_layer, "n_embd": ckpt.n_embd,
        "head_size": ckpt.head_size, "vocab_size": ckpt.vocab_size,
        "tensors": {},
    }
    n_sb6 = 0
    for key, qt in ckpt.tensors.items():
        if qt.gw_mode != "sb6":
            continue
        gw = GwQuantLinear(qt)
        assert gw._k3, f"{key}: K3-интерлив не построился (OUT%16 != 0?)"
        tensors[f"{key}::qblk"] = gw.qblk
        tensors[f"{key}::qsqm"] = gw.qsqm
        tensors[f"{key}::ddm"] = gw.ddm
        manifest["tensors"][key] = {
            "shape": list(qt.shape), "bits": qt.bits, "xbits": gw.xbits,
            "gw_gs": qt.gw_gs, "gw_sb": qt.gw_sb,
        }
        n_sb6 += 1

    mx.save_safetensors(out_path, tensors)
    with open(out_path + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"exported {n_sb6} sb6-тензоров -> {out_path} (+{out_path}.json)")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
