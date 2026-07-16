"""
То же сравнение canonical RTN vs SpQR (proj/cmix, INT4, outlier_frac=1%),
что уже прогонялось на 1.5B (WKV-kvant/test_sparse_outlier.log), но на
ru60m -- для честного сравнения ПОВЕДЕНИЯ ПРИ МАСШТАБИРОВАНИИ, не только
абсолютных чисел. Тот же corpus/batching паттерн, что в WKV-kvant/ablation.py.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

DEVICE = "cpu"
DTYPE = torch.float32
CKPT = os.path.expanduser("~/Develop/WKV-kvant/ru60m")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus.pt")

print("=== загрузка ru60m и корпуса ===")
model = RWKV7Ref(CKPT, device=DEVICE, dtype=DTYPE)
data = torch.load(CORPUS).to(DEVICE)
print(f"корпус: {data.shape}")


@torch.no_grad()
def perplexity(cfg: QuantConfig, batch_size=6):
    total_nll, total_tok = 0.0, 0
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i + batch_size]
        logits = model.forward(batch[:, :-1], cfg)
        target = batch[:, 1:]
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()
        total_tok += nll.numel()
    return torch.exp(torch.tensor(total_nll / total_tok)).item()


t0 = time.time()
baseline = perplexity(QuantConfig())
print(f"\nBASELINE bf16(fake fp32)  ppl={baseline:.4f}  [{time.time()-t0:.1f}s]")

configs = [
    ("proj  INT4 canonical RTN", QuantConfig(proj=4)),
    ("proj  INT4 SpQR frac=1%", QuantConfig(proj=4, outlier_fracs={"proj": 0.01})),
    ("cmix  INT4 canonical RTN", QuantConfig(cmix=4)),
    ("cmix  INT4 SpQR frac=1%", QuantConfig(cmix=4, outlier_fracs={"cmix": 0.01})),
]

results = {}
for name, cfg in configs:
    t0 = time.time()
    ppl = perplexity(cfg)
    delta = 100 * (ppl - baseline) / baseline
    print(f"{name:28s} ppl={ppl:10.4f}  Δ={delta:+8.2f}%  [{time.time()-t0:.1f}s]")
    results[name] = (ppl, delta)

print("\n=== СВОДКА: ru60m (61.3M) vs 1.5B (rwkv7-g1h), тот же protocol ===")
print(f"{'':30s}{'ru60m Δppl':>14s}{'1.5B Δppl':>14s}{'ratio (1.5B/ru60m)':>22s}")
ref_1p5b = {
    "proj  INT4 canonical RTN": 19.97,
    "proj  INT4 SpQR frac=1%": 5.62,
    "cmix  INT4 canonical RTN": 48.09,
    "cmix  INT4 SpQR frac=1%": 13.00,
}
for name, _ in configs:
    d60 = results[name][1]
    d15 = ref_1p5b[name]
    ratio = (d15 / d60) if abs(d60) > 1e-6 else float("inf")
    print(f"{name:30s}{d60:13.2f}%{d15:13.2f}%{ratio:21.2f}x")
