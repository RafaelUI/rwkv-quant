"""Бит-точность реального формата v2 против fake-dequant-пути.

Для каждого класса тензоров чемпион-конфига (proj@5, cmix@4, emb_head@5,
LoRA@6 gw64) квантуем один живой тензор 1.5B двумя путями:
  fake: quantize_tensor(real_gw=False) -> dense bf16
  real: quantize_tensor(real_gw=True)  -> packed -> reader._dequantize_one
и требуем ПОБИТОВОГО равенства bf16. Один процесс, без модели -- быстро."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import _dequantize_one

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")

CHAMPION = QuantConfig(
    proj=5, cmix=4, emb_head=5,
    w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=8,
    outlier_fracs={},
    group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                 "w_lora": 64, "a_lora": 64, "v_lora": 64},
    group_scale_mode={"proj": "asym_sb6_aw", "cmix": "asym_sb6_aw",
                      "emb_head": "asym_sb6_aw"},
    act_stats_path="/tmp/act_stats_1p5b.pt",
)


def pick_keys(sd):
    picked, seen = [], set()
    for k, w in sd.items():
        qt_probe = None
        for cls, pats in [
            ("proj", (".att.receptance.weight",)),
            ("cmix_key", (".ffn.key.weight",)),
            ("cmix_val", (".ffn.value.weight",)),
            ("head", ("head.weight",)),
            ("lora", (".w1", ".w_lora_A.weight")),
        ]:
            if cls not in seen and any(p in k for p in pats) and w.dim() == 2:
                picked.append((cls, k)); seen.add(cls)
    return picked


def main():
    sd = torch.load(CKPT, map_location="cpu")
    for cls, key in pick_keys(sd):
        w = sd[key]
        fake = quantize_tensor(key, w, CHAMPION, real_gw=False)
        real = quantize_tensor(key, w, CHAMPION, real_gw=True)
        assert fake.bits == 16 and fake.dense is not None, (cls, key)
        assert real.gw_mode in ("sb6", "asym"), (cls, key, real.gw_mode, real.bits)
        deq = _dequantize_one(real)
        same = torch.equal(fake.dense, deq)
        md = (fake.dense.float() - deq.float()).abs().max().item()
        nbytes = sum(t.numel() * t.element_size()
                     for t in (real.codes_packed, real.codes, real.gw_d, real.gw_dm,
                               real.gw_qsqm, real.gw_qh, real.gw_scale, real.gw_min)
                     if t is not None)
        bpw = nbytes * 8 / w.numel()
        print(f"{cls:9s} {key:38s} {str(tuple(w.shape)):16s} "
              f"mode={real.gw_mode:4s} bits={real.bits} bpw={bpw:5.3f} "
              f"bitexact={same} maxdiff={md:.3e}")
        assert same, f"MISMATCH {cls} {key}"
    print("ALL BIT-EXACT")


if __name__ == "__main__":
    main()
