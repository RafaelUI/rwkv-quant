"""ЭТАП 1 АВТОПОДБОРА (21.09): честная ошибка впрыска по тензорам.

Диагональное приближение (E[x^2]-взвешенная ошибка весов) ПРОВАЛИЛОСЬ: у
слоя 0 оно наименьшее во всех моделях, а вред максимальный. Здесь ошибка
считается на НАСТОЯЩИХ активациях, подвыборкой строк токенов:
    rel = ||(W - Ŵ) X|| / ||W X||
Никакой независимости каналов не предполагается. Вход берётся тем же
крюком, которым собирается act-статистика.
    python inject.py <чекпоинт> <act_stats.pt> [строк]
"""
import copy, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # корень репо: Mac и сервер
import torch
from rwkv_quant.presets import PRESETS
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import dequantize_banded
from rwkv_quant.models import rwkv7_ref as ref_mod
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.calibration import act_stats as A

ck, ap = sys.argv[1], sys.argv[2]
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
dev = os.environ.get("RWKVQ_DEVICE", "cuda")

# крюк: вместо сумм x^2 копим подвыборку строк
STORE = {}
class Rec(dict):
    def __setitem__(self, k, v):  # не используется, оставлено для совместимости
        dict.__setitem__(self, k, v)

orig = ref_mod._rec if hasattr(ref_mod, "_rec") else None
def rec(key, x):
    if key not in STORE:
        xs = x.reshape(-1, x.shape[-1])
        idx = torch.randperm(xs.shape[0], device=xs.device)[:ROWS]
        STORE[key] = xs[idx].float().cpu()
ref_mod._rec = rec
ref_mod.ACT_RECORDER = {}

model = RWKV7Ref(ck, device=dev, dtype=torch.bfloat16)
data, _ = A._windows_tokens(ck, ap) if hasattr(A, "_windows_tokens") else (None, None)
if data is None:
    import numpy as np
    blob = torch.load(os.path.expanduser("~/rwkvq/eval_corpus_multiling.pt"))
    tok = blob["tokens"] if isinstance(blob, dict) else blob
    data = tok[:4, :512].to(dev)
with torch.no_grad():
    for i in range(data.shape[0]):
        model.forward(data[i:i + 1, :-1])
del model
sd = torch.load(ck, map_location="cpu", mmap=True)
cfg = copy.deepcopy(PRESETS["compression"])
cfg.bits_overrides = {k: v for k, v in cfg.bits_overrides.items() if "blocks.0." not in k}
cfg.act_stats_path = ap
rows = []
for k, X in STORE.items():
    if k not in sd or sd[k].dim() < 2:
        continue
    W = sd[k].float()
    if X.shape[1] != W.shape[1]:
        continue
    q = quantize_tensor(k, sd[k], cfg, real_gw=True)
    if q.bits >= 16:
        continue
    dW = dequantize_banded(q, dtype=torch.float32) - W
    num = (X @ dW.T).norm().item(); den = (X @ W.T).norm().item()
    rows.append((num / max(den, 1e-30), k))
rows.sort(reverse=True)
print("ТОП-12 по честной ошибке впрыска  ||(W-Ŵ)X|| / ||WX||")
for r, k in rows[:12]:
    print("  %.4f  %s" % (r, k))
print("медиана %.4f, всего тензоров %d" % (sorted(r for r, _ in rows)[len(rows) // 2], len(rows)))
json.dump([[r, k] for r, k in rows], open(os.path.expanduser("~/rwkvq/inject_%s.json" % os.path.basename(ck)[:20]), "w"))
