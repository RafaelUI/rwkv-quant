"""Гейт: RWKV7Ref с bf16-ХРАНЕНИЕМ и fp32-СЧЁТОМ численно равен fp32-модели (22.09).
На этом стоит measure (autopick): 2.9B в 5.9 ГБ вместо 11.8 при том же приборе.
Утверждения, на CPU и (если есть) на MPS, 0.1B, два окна по 64 токена:
  1  bf16+compute fp32 == fp32 побитно (апкаст bf16->fp32 точен);
  2  forward(return_hidden) @ head == forward побитно;
  3  РАЗРЕШЕНИЕ: bf16-счёт != fp32 (иначе утверждение 1 ничего не проверяет);
  4  fake-подмена одной матрицы через calibration.q в fp32 проходит (тип счёта);
  5  с квантующим конфигом (COMPRESSION) bf16+fp32 == fp32 побитно.
    python tests/test_ref_storage_dtype.py [чекпоинт]"""
import copy, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.calibration import fake_quant
from rwkv_quant import presets
CK = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
idx = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))["tokens"][[0, 20], :64].contiguous()
devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
fails = 0
for dev in devs:
    i = idx.to(dev)
    with torch.no_grad():
        F32 = RWKV7Ref(CK, device=dev, dtype=torch.float32)
        l32 = F32.forward(i); del F32
        B16 = RWKV7Ref(CK, device=dev, dtype=torch.bfloat16)
        l16 = B16.forward(i).float(); del B16
        M = RWKV7Ref(CK, device=dev, dtype=torch.bfloat16, compute_dtype=torch.float32)
        lm = M.forward(i)
        h = M.forward(i, return_hidden=True)
        lh = h @ M.head_weight.float().T
        # 4: одна матрица квантована в fp32, остальное bf16-хранение
        key = "blocks.3.ffn.value.weight"; c = M.cmix[3]; w = c.value
        cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = None
        c.value = fake_quant.q(w.float(), "cmix", cfg, key)
        lq = M.forward(i); c.value = w
        lb = M.forward(i)
        # 5: квантующий конфиг (весь COMPRESSION, без AW) -- bf16+fp32 == fp32 побитно
        cq = copy.deepcopy(presets.COMPRESSION); cq.act_stats_path = None
        lq_m = M.forward(i, cfg=cq); del M
        F32 = RWKV7Ref(CK, device=dev, dtype=torch.float32)
        lq_32 = F32.forward(i, cfg=cq); del F32
    a5 = torch.equal(lq_m, lq_32) and not torch.equal(lq_m, lm)
    a1 = torch.equal(lm, l32); a2 = torch.equal(lh, lm)
    a3 = not torch.equal(l16, l32); d3 = float((l16 - l32).abs().max())
    a4 = lq.dtype == torch.float32 and not torch.equal(lq, lm) and torch.equal(lb, lm)
    print("%-4s 1 bf16+fp32==fp32 %s | 2 hidden@head==forward %s | 3 bf16-счёт отличим %s (max|d| %.3g) | 4 подмена fp32 %s | 5 квант. конфиг ==fp32 %s"
          % (dev, a1, a2, a3, d3, a4, a5))
    fails += not (a1 and a2 and a3 and a4 and a5)
print("ИТОГ:", "ЗЕЛЁНЫЙ" if not fails else "КРАСНЫЙ")
sys.exit(1 if fails else 0)
