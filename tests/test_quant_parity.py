"""
Численная сверка backends/metal/quant_model.py (MLX, реальное int8+SpQR
квантование) против models/rwkv7_ref.py (torch, fake-quant тем же cfg).

fake_quantize_sparse_outlier (calibration/fake_quant.py) и
_real_quantize_sparse_outlier (formats/writer.py) используют ОДНУ И ТУ ЖЕ
формулу (top-k outlier по строке, RTN-скейл на остатке) -- разница только
в том, что первая держит результат как float, а вторая паковает в int8
codes + fp16 scale. Поэтому если оба пути дают близкие логиты -- значит
quant_model.py корректно воспроизводит архитектуру RWKV7Ref, а не только
"не падает", как в smoke-тесте.

RWKV7Ref гоняется на cpu/float32 (не mps/bfloat16 по умолчанию), чтобы не
подмешивать сюда ещё и bf16-шум -- он уже отдельно измерен в
test_quant_linear_metal.py и не является багом.
"""
import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import mlx.core as mx
from safetensors.torch import save_file

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import save as save_rwkvq
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

torch.manual_seed(0)

N_LAYER, D, HEAD_SIZE, VOCAB = 2, 128, 64, 256
H = D // HEAD_SIZE
CKPT_DIR = "/tmp/toy_ckpt_dir"
RWKVQ_PATH = "/tmp/toy_parity.rwkvq"


def _lora_ranks(d):
    f = lambda c, p: max(32, int(round((c * (d ** p)) / 32) * 32))
    return {"w": f(1.8, 0.5), "a": f(1.8, 0.5), "v": f(1.3, 0.5), "g": f(0.6, 0.8)}


def build_toy_state_dict():
    r = _lora_ranks(D)
    sd = {}
    sd["emb.weight"] = torch.randn(VOCAB, D) * 0.02
    sd["head.weight"] = torch.randn(VOCAB, D) * 0.02
    sd["ln0.weight"] = torch.ones(D); sd["ln0.bias"] = torch.zeros(D)
    sd["ln_out.weight"] = torch.ones(D); sd["ln_out.bias"] = torch.zeros(D)

    for i in range(N_LAYER):
        p = f"blocks.{i}."
        sd[p+"ln1.weight"] = torch.ones(D); sd[p+"ln1.bias"] = torch.zeros(D)
        sd[p+"ln2.weight"] = torch.ones(D); sd[p+"ln2.bias"] = torch.zeros(D)

        tp = p + "tmix."
        for nm in ("x_r","x_w","x_k","x_v","x_a","x_g"):
            sd[tp+nm] = torch.rand(1,1,D) * 0.6 + 0.2  # избегаем 0/1 краёв
        sd[tp+"w_lora_A.weight"] = torch.randn(r["w"], D) * 0.02
        sd[tp+"w_lora_B.weight"] = torch.randn(D, r["w"]) * 0.02
        sd[tp+"w_lora_B.bias"] = torch.randn(D) * 0.1
        sd[tp+"a_lora_A.weight"] = torch.randn(r["a"], D) * 0.02
        sd[tp+"a_lora_B.weight"] = torch.randn(D, r["a"]) * 0.02
        sd[tp+"a_lora_B.bias"] = torch.randn(D) * 0.1
        if i > 0:
            sd[tp+"v_lora_A.weight"] = torch.randn(r["v"], D) * 0.02
            sd[tp+"v_lora_B.weight"] = torch.randn(D, r["v"]) * 0.02
            sd[tp+"v_lora_B.bias"] = torch.randn(D) * 0.1
        sd[tp+"g_lora_A.weight"] = torch.randn(r["g"], D) * 0.02
        sd[tp+"g_lora_B.weight"] = torch.randn(D, r["g"]) * 0.02
        sd[tp+"k_k"] = torch.rand(H, HEAD_SIZE) * 0.5 + 0.5
        sd[tp+"k_a"] = torch.rand(H, HEAD_SIZE) * 0.5
        sd[tp+"r_k"] = torch.randn(H, HEAD_SIZE) * 0.1
        sd[tp+"r_proj.weight"] = torch.randn(D, D) * 0.02
        sd[tp+"k_proj.weight"] = torch.randn(D, D) * 0.02
        sd[tp+"v_proj.weight"] = torch.randn(D, D) * 0.02
        sd[tp+"o_proj.weight"] = torch.randn(D, D) * 0.02
        # инжектим outlier'ы, как в предыдущих тестах -- иначе SpQR-ветка молчит
        for row in range(0, D, 5):
            col = np.random.randint(0, D)
            sd[tp+"r_proj.weight"][row, col] *= 50.0
            sd[tp+"k_proj.weight"][row, col] *= 50.0
        sd[tp+"ln_x.weight"] = torch.ones(D); sd[tp+"ln_x.bias"] = torch.zeros(D)

        cp = p + "cmix."
        sd[cp+"x_k"] = torch.rand(1,1,D) * 0.6 + 0.2
        sd[cp+"key.weight"] = torch.randn(D*4, D) * 0.02
        sd[cp+"value.weight"] = torch.randn(D, D*4) * 0.02
        for row in range(0, D*4, 5):
            col = np.random.randint(0, D)
            sd[cp+"key.weight"][row, col] *= 50.0

    return sd


def main():
    sd = build_toy_state_dict()
    cfg = QuantConfig(proj=8, cmix=8, outlier_fracs={"proj": 0.02, "cmix": 0.01})

    # --- реальное квантование -> .rwkvq -> QuantRWKV7 (MLX) ---
    save_rwkvq(sd, cfg, RWKVQ_PATH, naming="custom", n_layer=N_LAYER, n_embd=D,
               head_size=HEAD_SIZE, vocab_size=VOCAB)
    ckpt = load_raw(RWKVQ_PATH)
    quant_model = QuantRWKV7(ckpt)

    # --- тот же sd на диск как custom-naming чекпоинт для RWKV7Ref ---
    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR)
    save_file(sd, f"{CKPT_DIR}/model.safetensors")
    with open(f"{CKPT_DIR}/config.json", "w") as f:
        json.dump({"n_layer": N_LAYER, "n_embd": D, "head_size": HEAD_SIZE, "vocab_size": VOCAB}, f)

    ref_model = RWKV7Ref(CKPT_DIR, device="cpu", dtype=torch.float32)

    idx_list = [1, 5, 17, 42, 100, 3, 9, 200]
    idx_mx = mx.array([idx_list])
    idx_torch = torch.tensor([idx_list])

    logits_quant = np.array(quant_model(idx_mx))
    with torch.no_grad():
        logits_ref = ref_model(idx_torch, cfg=cfg).numpy()

    print(f"quant  logits: shape={logits_quant.shape} mean={logits_quant.mean():.4f} std={logits_quant.std():.4f}")
    print(f"ref    logits: shape={logits_ref.shape} mean={logits_ref.mean():.4f} std={logits_ref.std():.4f}")

    abs_err = np.abs(logits_quant - logits_ref)
    rel_err = abs_err.max() / (np.abs(logits_ref).max() + 1e-8)
    print(f"max abs err: {abs_err.max():.6f}   mean abs err: {abs_err.mean():.6f}   max rel err: {rel_err:.6e}")

    # argmax по последней позиции -- практический "не сломали генерацию" сигнал
    top_quant = logits_quant[0, -1].argsort()[-5:]
    top_ref = logits_ref[0, -1].argsort()[-5:]
    overlap = len(set(top_quant.tolist()) & set(top_ref.tolist()))
    print(f"top-5 next-token overlap: {overlap}/5")

    ok = rel_err < 1e-2  # допуск: fp32 recurrence на mlx vs torch, разные BLAS -- не бит-в-бит
    print(f"\n[{'OK' if ok else 'FAIL'}] parity within tolerance: {ok}")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
