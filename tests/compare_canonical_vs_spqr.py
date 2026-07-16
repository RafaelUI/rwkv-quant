"""
Сравнение "канонического" квантования (plain per-row RTN, formats.writer.
_real_quantize -- то, что делает почти любой стандартный INT8 quantizer,
bitsandbytes включая) против SpQR-подхода этого проекта
(_real_quantize_sparse_outlier) -- на весах с инжектированными outlier'ами
(как в r_proj/k_proj/key.weight здесь и как реально наблюдалось на 1.5B
в README: per-channel outliers 40-96x).

Два среза:
  1. Reconstruction error весов напрямую (||W - dequant(W)||) -- изолирует
     эффект метода квантования от архитектуры.
  2. End-to-end: логиты полной модели (через RWKV7Ref + fake_quant с тем
     же cfg) против fp32 baseline -- показывает, доходит ли ошибка
     реконструкции до выхода модели или гасится по пути.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.fake_quant import q
from rwkv_quant.formats.writer import _real_quantize, _real_quantize_sparse_outlier
from rwkv_quant.formats.reader import _dequantize_one
from rwkv_quant.formats.schema import QuantizedTensor
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

torch.manual_seed(0)
np.random.seed(0)


def weight_level_comparison():
    print("=" * 70)
    print("1. RECONSTRUCTION ERROR (веса с инжектированными outlier'ами)")
    print("=" * 70)
    D, BITS, FRAC = 512, 8, 0.02
    w = torch.randn(D, D) * 0.02
    for row in range(0, D, 5):
        col = np.random.randint(0, D)
        w[row, col] *= 60.0  # как реальные per-channel outliers из README

    codes, scale = _real_quantize(w, BITS)
    qt_rtn = QuantizedTensor(key="w", group="proj", bits=BITS, shape=tuple(w.shape),
                              codes=codes, scale=scale)
    w_rtn = _dequantize_one(qt_rtn).float()

    codes, scale, oi, ov = _real_quantize_sparse_outlier(w, BITS, FRAC)
    qt_spqr = QuantizedTensor(key="w", group="proj", bits=BITS, shape=tuple(w.shape),
                               codes=codes, scale=scale, outlier_indices=oi, outlier_values=ov)
    w_spqr = _dequantize_one(qt_spqr).float()

    err_rtn = (w - w_rtn).abs()
    err_spqr = (w - w_spqr).abs()

    print(f"canonical RTN:  mean abs err={err_rtn.mean():.5f}  max abs err={err_rtn.max():.5f}  "
          f"frob rel err={(err_rtn.norm()/w.norm()):.5f}")
    print(f"SpQR (проект):  mean abs err={err_spqr.mean():.5f}  max abs err={err_spqr.max():.5f}  "
          f"frob rel err={(err_spqr.norm()/w.norm()):.5f}")
    print(f"-> RTN ошибка в {(err_rtn.norm()/err_spqr.norm()):.1f}x больше по Frobenius-норме\n")


def end_to_end_comparison():
    print("=" * 70)
    print("2. END-TO-END: логиты против fp32 baseline")
    print("=" * 70)
    from tests.test_quant_parity import build_toy_state_dict, N_LAYER, D, HEAD_SIZE, VOCAB, CKPT_DIR
    import json, shutil
    from safetensors.torch import save_file

    sd = build_toy_state_dict()
    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR)
    save_file(sd, f"{CKPT_DIR}/model.safetensors")
    with open(f"{CKPT_DIR}/config.json", "w") as f:
        json.dump({"n_layer": N_LAYER, "n_embd": D, "head_size": HEAD_SIZE, "vocab_size": VOCAB}, f)

    model = RWKV7Ref(CKPT_DIR, device="cpu", dtype=torch.float32)

    cfg_fp32 = QuantConfig()  # bits=16 везде -> без квантования
    cfg_rtn = QuantConfig(proj=8, cmix=8)  # каноническое RTN, без outlier_fracs
    cfg_spqr = QuantConfig(proj=8, cmix=8, outlier_fracs={"proj": 0.02, "cmix": 0.01})

    torch.manual_seed(1)
    idx = torch.randint(0, VOCAB, (4, 16))  # несколько случайных последовательностей

    with torch.no_grad():
        logits_fp32 = model(idx, cfg=cfg_fp32)
        logits_rtn = model(idx, cfg=cfg_rtn)
        logits_spqr = model(idx, cfg=cfg_spqr)

    def stats(name, logits):
        err = (logits - logits_fp32).abs()
        # top-1 agreement по каждой позиции с baseline
        agree = (logits.argmax(-1) == logits_fp32.argmax(-1)).float().mean().item()
        # KL(baseline || quantized) как прокси для ppl-эффекta
        p = torch.softmax(logits_fp32, dim=-1)
        logq = torch.log_softmax(logits, dim=-1)
        logp = torch.log_softmax(logits_fp32, dim=-1)
        kl = (p * (logp - logq)).sum(-1).mean().item()
        print(f"{name:22s} mean_abs_err={err.mean():.5f}  max_abs_err={err.max():.5f}  "
              f"top1_agree={agree:.3f}  KL(fp32||q)={kl:.5f}")

    stats("canonical RTN:", logits_rtn)
    stats("SpQR (проект):", logits_spqr)


if __name__ == "__main__":
    weight_level_comparison()
    end_to_end_comparison()
