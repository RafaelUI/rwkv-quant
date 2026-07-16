"""
Проверка streaming-инференса (forward_stateful + wkv7_infer chunked,
no-op padding) на игрушечном чекпоинте:

  1. forward_stateful(idx, init_state) на весь промпт одним вызовом ==
     __call__(idx) (wkv7_train, не-streaming путь) -- проверяет, что
     no-op padding трюк не портит математику при чанковании.
  2. Тот же промпт, но token-by-token (T=1 за вызов, state переносится
     между вызовами) == тот же forward_stateful одним вызовом на весь
     промпт -- проверяет, что state корректно живёт между вызовами
     (это и есть весь смысл streaming decode).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import mlx.core as mx

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import save as save_rwkvq
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from tests.test_quant_model_smoke import build_toy_state_dict, N_LAYER, D, HEAD_SIZE, VOCAB

torch.manual_seed(0)

RWKVQ_PATH = "/tmp/toy_streaming.rwkvq"


def main():
    sd = build_toy_state_dict()
    cfg = QuantConfig(proj=8, cmix=8, outlier_fracs={"proj": 0.02, "cmix": 0.01})
    save_rwkvq(sd, cfg, RWKVQ_PATH, naming="custom", n_layer=N_LAYER, n_embd=D,
               head_size=HEAD_SIZE, vocab_size=VOCAB)
    ckpt = load_raw(RWKVQ_PATH)
    model = QuantRWKV7(ckpt)

    # T=40 нарочно (> CHUNK=32), чтобы задействовать и полный чанк, и хвост
    tokens = [1, 5, 17, 42, 100, 3, 9, 200, 7, 88, 15, 33, 91, 4, 62, 19,
              23, 71, 8, 45, 2, 66, 12, 90, 34, 55, 21, 6, 77, 40,
              29, 11, 60, 3, 99, 18, 27, 5, 84, 50]
    assert len(tokens) == 40
    idx_full = mx.array([tokens])

    # --- эталон: не-streaming forward (wkv7_train) ---
    logits_ref = np.array(model(idx_full))

    # --- 1) forward_stateful, весь промпт одним вызовом ---
    states0 = model.init_state(batch_size=1)
    logits_batched, _ = model.forward_stateful(idx_full, states0)
    logits_batched = np.array(logits_batched)

    err1 = np.abs(logits_ref - logits_batched).max()
    rel1 = err1 / (np.abs(logits_ref).max() + 1e-8)
    print(f"[batched-stateful vs wkv7_train]  max_abs_err={err1:.6f}  rel_err={rel1:.6e}")

    # --- 2) token-by-token, state переносится между вызовами ---
    states = model.init_state(batch_size=1)
    logits_stream = []
    for t in tokens:
        idx_t = mx.array([[t]])
        logits_t, states = model.forward_stateful(idx_t, states)
        logits_stream.append(np.array(logits_t)[0, 0])
    logits_stream = np.stack(logits_stream)[None, :, :]  # [1, T, VOCAB]

    err2 = np.abs(logits_ref - logits_stream).max()
    rel2 = err2 / (np.abs(logits_ref).max() + 1e-8)
    print(f"[token-by-token vs wkv7_train]    max_abs_err={err2:.6f}  rel_err={rel2:.6e}")

    top_ref = logits_ref[0, -1].argsort()[-5:]
    top_stream = logits_stream[0, -1].argsort()[-5:]
    overlap = len(set(top_ref.tolist()) & set(top_stream.tolist()))
    print(f"top-5 next-token overlap (last position): {overlap}/5")

    # Допуск 1e-3 (был 1e-4 под fp32-dense): с fp16-хранением dense/LoRA
    # (см. quant_model._dense) fp16-округление накапливается через
    # рекуррентность по шагам token-by-token и даёт rel ~2.6e-4 против
    # батчевого прохода. Проверено на 1.5B REDUCTION: ppl fp16 vs fp32
    # идентичен (11.5186 vs 11.5191), top-5 совпадает -- это шум формата,
    # не деградация. Расхождение батчевого пути осталось нулевым (rel1).
    ok = rel1 < 1e-4 and rel2 < 1e-3
    print(f"\n[{'OK' if ok else 'FAIL'}] streaming inference: state carries correctly across calls")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
