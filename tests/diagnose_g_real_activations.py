"""То же самое, что g_branch_test() в diagnose_lora_orientation.py, но с
РЕАЛЬНЫМИ активациями xg (из настоящего forward pass на eval-корпусе),
а не iid Gaussian. Цель: понять, закрывает ли это разрыв между синтетическим
тестом (writer.py хуже ref всего в ~1.5-1.7x) и реальным ppl-разрывом (13x)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.formats.writer import _real_quantize
from rwkv_quant.calibration.group_config import QuantConfig

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
LAYER = 1
BITS = 4


def dequant(codes, scale):
    return codes.float() * scale.float()


@torch.no_grad()
def get_real_xg(model, idx):
    """Копия начала forward() до нужного слоя, чтобы вытащить xg = x + (shift(x)-x)*x_g
    на слое LAYER, без квантования (полная точность, baseline-путь)."""
    x = F.embedding(idx, model.emb_weight)
    x = F.layer_norm(x.float(), (model.n_embd,), model.ln0_w.float(), model.ln0_b.float()).to(x.dtype)
    v_first = torch.empty_like(x)
    for i in range(LAYER + 1):
        xn = F.layer_norm(x.float(), (model.n_embd,), model.ln1_w[i].float(), model.ln1_b[i].float()).to(x.dtype)
        if i == LAYER:
            t = model.tmix[i]
            xx = model._time_shift(xn) - xn
            xg = xn + xx * t.x_g
            return xg.reshape(-1, model.n_embd).float()  # [B*T, C]
        att, v_first = model._tmix_forward(xn, v_first, model.tmix[i], i, QuantConfig())
        x = x + att
        xn2 = F.layer_norm(x.float(), (model.n_embd,), model.ln2_w[i].float(), model.ln2_b[i].float()).to(x.dtype)
        x = x + model._cmix_forward(xn2, model.cmix[i], QuantConfig())


def main():
    model = RWKV7Ref(CKPT_PTH, device="cpu", dtype=torch.bfloat16)
    data = torch.load(CORPUS)[:4]  # немного токенов достаточно
    idx = data[:, :512]  # первые 512 токенов каждой последовательности

    xg_real = get_real_xg(model, idx)
    print(f"real xg stats: shape={tuple(xg_real.shape)} mean={xg_real.mean():.4f} "
          f"std={xg_real.std():.4f} absmax={xg_real.abs().max():.4f}")

    x_synth = torch.randn(xg_real.shape[0], xg_real.shape[1])
    print(f"synth x stats: shape={tuple(x_synth.shape)} mean={x_synth.mean():.4f} "
          f"std={x_synth.std():.4f} absmax={x_synth.abs().max():.4f}")

    sd = torch.load(CKPT_PTH, map_location="cpu")
    p = f"blocks.{LAYER}.att."
    g1 = sd[p + "g1"].float()
    g2 = sd[p + "g2"].float()

    def forward_raw(x, g1_, g2_):
        h = torch.sigmoid(x @ g1_)
        return h @ g2_

    for label, x in [("real xg", xg_real), ("synthetic randn", x_synth)]:
        y_true = forward_raw(x, g1, g2)

        c1, s1 = _real_quantize(g1, BITS); g1_hat_raw = dequant(c1, s1)
        c2, s2 = _real_quantize(g2, BITS); g2_hat_raw = dequant(c2, s2)
        y_writer = forward_raw(x, g1_hat_raw, g2_hat_raw)

        c1t, s1t = _real_quantize(g1.T.contiguous(), BITS); g1_hat_T = dequant(c1t, s1t).T.contiguous()
        c2t, s2t = _real_quantize(g2.T.contiguous(), BITS); g2_hat_T = dequant(c2t, s2t).T.contiguous()
        y_ref = forward_raw(x, g1_hat_T, g2_hat_T)

        err_writer = (y_true - y_writer).norm().item() / y_true.norm().item()
        err_ref = (y_true - y_ref).norm().item() / y_true.norm().item()
        max_writer = (y_true - y_writer).abs().max().item()
        max_ref = (y_true - y_ref).abs().max().item()
        print(f"[{label}] rel err: writer={err_writer:.6f} ref={err_ref:.6f} ratio={err_writer/max(err_ref,1e-12):.3f} | "
              f"max err: writer={max_writer:.6f} ref={max_ref:.6f} ratio={max_writer/max(max_ref,1e-12):.3f}")


if __name__ == "__main__":
    main()
