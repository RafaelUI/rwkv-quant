"""A/B: старая (uint8-распакованные qs/qm) vs новая (упакованные 6-бит,
распаковка в кернеле) реализация GwQuantLinear. Один процесс, чередование,
sync амортизирован (закон 19.07-2/3) -- методология как bench_kernel_clean.py.

РЕЗУЛЬТАТ (сессия 19.07-4): NEW медленнее OLD на 3-6% (не быстрее) --
резерв "-0.125 bpw трафика" оказался мнимым, лишние инструкции в кернеле
(6 байт-чтений + select() вместо 2 скалярных чтений) дороже сэкономленных
байт. rwkv_quant/backends/metal/quant_linear_gw.py ОТКАЧЕН обратно к
uint8-версии. Скрипт оставлен как свидетельство отрицательного результата.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, mlx.core as mx
from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear as GwNEW
from _gw_old_for_ab import GwQuantLinearOLD as GwOLD
from rwkv_quant.formats.writer import quantize_tensor
# NB: полный QuantRWKV7(champion_v2.rwkvq) сюда НЕ грузим -- он не нужен
# (сравнение идёт на 5 отдельно квантованных тензорах ниже), а лишняя
# загрузка ~1GB packed + torch state_dict 2.8GB одновременно на 16GB
# машине однажды увела в своп. Держим только то, что реально используется.
import torch
from test_v2_format import CHAMPION
CKPT_PATH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)

keys = ["blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight",
        "blocks.0.att.receptance.weight", "blocks.0.att.key.weight",
        "head.weight"]
old_lins, newq_lins, sizes_mb = [], [], []
for k in keys:
    qt = quantize_tensor(k, sd[k], CHAMPION, real_gw=True)
    old_lins.append(GwOLD(qt))
    newq_lins.append(GwNEW(qt))
    OUT, IN = qt.shape
    sizes_mb.append(OUT * IN * 4.5 / 8 / 1e6)  # ~bpw в файле, для справки

xs = {}
for k in keys:
    IN = sd[k].shape[1]
    if IN not in xs:
        xa = mx.array(np.random.randn(1, IN).astype(np.float32)); mx.eval(xa)
        xs[IN] = xa
x_for = [xs[sd[k].shape[1]] for k in keys]

def bench(fn, reps=15, warm=5):
    for _ in range(warm): mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); mx.eval(*fn()); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))

def run_old(): return [l(xi) for l, xi in zip(old_lins, x_for)]
def run_new(): return [l(xi) for l, xi in zip(newq_lins, x_for)]
def sync_only(): return [xs[2048] + 1.0]

print("предварительный разогрев DVFS (2с)...")
_a = mx.ones((2048, 2048), dtype=mx.float16)
_t0 = time.perf_counter()
while time.perf_counter() - _t0 < 2.0: mx.eval(_a @ _a)
mx.synchronize()

acc = {"old": [], "new": [], "sync": []}
for _ in range(14):
    acc["old"].append(bench(run_old))
    acc["new"].append(bench(run_new))
    acc["sync"].append(bench(sync_only, reps=30))

# отбросить первые 2 внешних повтора (остаточный разогрев)
for k in acc: acc[k] = acc[k][2:]
sync = np.median(acc["sync"])
old_t = np.median(acc["old"]); new_t = np.median(acc["new"])
print(f"хост-синк:              {sync:6.3f} мс")
print(f"OLD (uint8 qs/qm):       {old_t:7.3f} мс -> {old_t-sync:6.3f} мс net")
print(f"NEW (packed 6-бит):      {new_t:7.3f} мс -> {new_t-sync:6.3f} мс net")
print(f"дельта: {(old_t-sync)-(new_t-sync):+.4f} мс "
      f"({((old_t-sync)-(new_t-sync))/(old_t-sync)*100:+.2f}%)")
print(f"все прогоны OLD: {[f'{v:.3f}' for v in acc['old']]}")
print(f"все прогоны NEW: {[f'{v:.3f}' for v in acc['new']]}")

# бит-точность: сверить NEW против OLD на этих же 5 тензорах
maxrel = 0.0
for lo, ln, k, xi in zip(old_lins, newq_lins, keys, x_for):
    yo = np.array(lo(xi)); yn = np.array(ln(xi))
    rel = np.abs(yo - yn).max() / (np.abs(yo).max() + 1e-9)
    maxrel = max(maxrel, rel)
    print(f"{k:34s} OLD vs NEW relmax={rel:.3e}")
print(f"max relmax OLD vs NEW: {maxrel:.3e}")
