"""
Smoke-тест backends/metal/quant_model.py: собираем игрушечный RWKV-7 x070
чекпоинт (custom naming, 2 слоя, D=128, head_size=64 -> H=2), реально
квантуем proj/cmix (int8 + SpQR), остальное dense, сохраняем в .rwkvq,
загружаем через QuantRWKV7 и прогоняем forward. Проверяем: не падает,
формы верные, логиты не NaN/Inf.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import mlx.core as mx

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import save
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

torch.manual_seed(0)

N_LAYER, D, HEAD_SIZE, VOCAB = 2, 128, 64, 256
H = D // HEAD_SIZE


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
            sd[tp+nm] = torch.zeros(1,1,D) + 0.5
        sd[tp+"w_lora_A.weight"] = torch.randn(r["w"], D) * 0.02
        sd[tp+"w_lora_B.weight"] = torch.randn(D, r["w"]) * 0.02
        sd[tp+"w_lora_B.bias"] = torch.zeros(D)
        sd[tp+"a_lora_A.weight"] = torch.randn(r["a"], D) * 0.02
        sd[tp+"a_lora_B.weight"] = torch.randn(D, r["a"]) * 0.02
        sd[tp+"a_lora_B.bias"] = torch.zeros(D)
        if i > 0:
            sd[tp+"v_lora_A.weight"] = torch.randn(r["v"], D) * 0.02
            sd[tp+"v_lora_B.weight"] = torch.randn(D, r["v"]) * 0.02
            sd[tp+"v_lora_B.bias"] = torch.zeros(D)
        sd[tp+"g_lora_A.weight"] = torch.randn(r["g"], D) * 0.02
        sd[tp+"g_lora_B.weight"] = torch.randn(D, r["g"]) * 0.02
        sd[tp+"k_k"] = torch.ones(H, HEAD_SIZE)
        sd[tp+"k_a"] = torch.zeros(H, HEAD_SIZE)
        sd[tp+"r_k"] = torch.zeros(H, HEAD_SIZE)
        sd[tp+"r_proj.weight"] = torch.randn(D, D) * 0.02
        sd[tp+"k_proj.weight"] = torch.randn(D, D) * 0.02
        sd[tp+"v_proj.weight"] = torch.randn(D, D) * 0.02
        sd[tp+"o_proj.weight"] = torch.randn(D, D) * 0.02
        sd[tp+"ln_x.weight"] = torch.ones(D); sd[tp+"ln_x.bias"] = torch.zeros(D)

        cp = p + "cmix."
        sd[cp+"x_k"] = torch.zeros(1,1,D) + 0.5
        sd[cp+"key.weight"] = torch.randn(D*4, D) * 0.02
        sd[cp+"value.weight"] = torch.randn(D, D*4) * 0.02

    return sd


def main():
    sd = build_toy_state_dict()
    cfg = QuantConfig(proj=8, cmix=8, outlier_fracs={"proj": 0.02, "cmix": 0.01})
    out_path = "/tmp/toy_model.rwkvq"
    save(sd, cfg, out_path, naming="custom", n_layer=N_LAYER, n_embd=D,
         head_size=HEAD_SIZE, vocab_size=VOCAB)
    print("saved .rwkvq ok")

    ckpt = load_raw(out_path)
    n_quant = sum(1 for qt in ckpt.tensors.values() if qt.bits < 16)
    n_spqr = sum(1 for qt in ckpt.tensors.values()
                 if qt.outlier_indices is not None and qt.outlier_indices.numel() > 0)
    print(f"quantized tensors: {n_quant}, with SpQR outliers: {n_spqr}")

    model = QuantRWKV7(ckpt)
    print(f"model built: n_layer={model.n_layer} n_embd={model.n_embd} "
          f"n_head={model.n_head} head_size={model.head_size} vocab={model.vocab_size}")

    idx = mx.array([[1, 5, 17, 42, 100, 3, 9, 200]])  # B=1, T=8
    logits = model(idx)
    mx.eval(logits)
    logits_np = __import__("numpy").array(logits)

    print(f"logits shape: {logits_np.shape}  (expected (1, 8, {VOCAB}))")
    assert logits_np.shape == (1, 8, VOCAB), "shape mismatch"

    has_nan = bool(__import__("numpy").isnan(logits_np).any())
    has_inf = bool(__import__("numpy").isinf(logits_np).any())
    print(f"NaN present: {has_nan}, Inf present: {has_inf}")
    print(f"logits stats: min={logits_np.min():.3f} max={logits_np.max():.3f} "
          f"mean={logits_np.mean():.3f} std={logits_np.std():.3f}")

    assert not has_nan and not has_inf, "NaN/Inf in output"
    print("\n[OK] smoke test passed")


if __name__ == "__main__":
    main()
