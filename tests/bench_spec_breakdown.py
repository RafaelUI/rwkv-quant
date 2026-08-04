"""
Почему спекулятивка с драфт-моделью оказалась 0.53x: разложение стоимости
раунда на составляющие.

Гипотеза: черновик 0.1B в 9.5 раз дешевле цели ПО ТРАФИКУ (92 МБ против
878 МБ за токен), но декод маленькой модели упирается не в полосу памяти,
а в задержку: 12 слоёв x десятки kernel launch плюс питонный цикл плюс
синхронизация GPU->CPU на каждом шаге (argmax -> .tolist()). Если шаг
черновика стоит не ~1 мс (сколько дают 92 МБ на 96 ГБ/с), а 8-10 мс, то
четыре шага черновика съедают больше, чем экономит верификация.

Меряется:
  - шаг черновика и шаг цели при T=1 (полный цикл с синхронизацией);
  - то же БЕЗ синхронизации (argmax остаётся на GPU) -- показывает, какая
    доля времени уходит на sync, а не на счёт;
  - шаг цели при T=1/5/9 -- во сколько реально обходится верификация
    нескольких токенов за проход.

Сравнение с полосой памяти показывает, где потолок, а где накладные.

    python tests/bench_spec_breakdown.py
"""
import copy
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mlx.core as mx  # noqa: E402

from rwkv_quant.presets import COMPRESSION  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

TARGET_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
DRAFT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
ACT_STATS = "/tmp/act_stats_1p5b_multiling.pt"
BW = 95.7          # ГБ/с, измеренная достижимая полоса на этом M4
R = 40


def tensor_bytes(qt):
    n = 0
    for f in ("dense", "codes", "codes_packed", "scale", "gw_qsqm", "gw_d",
              "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
              "outlier_indices", "outlier_values"):
        t = getattr(qt, f, None)
        if t is not None:
            n += t.numel() * t.element_size()
    return n


def build(pth, act_stats):
    sd = torch.load(pth, map_location="cpu", mmap=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    cfg = copy.deepcopy(COMPRESSION)
    cfg.act_stats_path = act_stats
    cfg.bits["small"] = 16
    tensors = {k: quantize_tensor(k, w, cfg, real_gw=True) for k, w in sd.items()}
    ckpt = QuantizedCheckpoint(
        naming="world", n_layer=n_layer, n_embd=sd["emb.weight"].shape[1],
        head_size=64, vocab_size=sd["emb.weight"].shape[0],
        tensors=tensors, config_repr=repr(cfg))
    model = QuantRWKV7(ckpt)
    mb = (sum(tensor_bytes(q) for q in tensors.values())
          - tensor_bytes(tensors["emb.weight"])) / 1e6
    del ckpt, tensors, sd
    gc.collect()
    return model, mb


def timed(model, T, sync_cpu):
    """sync_cpu=True -- как в реальном цикле: argmax уезжает на CPU через
    .tolist(), то есть каждый шаг это барьер GPU->CPU. False -- argmax
    остаётся массивом MLX, шаги конвейеризуются."""
    st = model.init_state(1)
    x = mx.array([[100] * max(T, 64)])
    _, st = model.step(x, st, tail_only=1)
    xt = mx.array([[100] * T])
    for _ in range(8):
        lg, _ = model.step(xt, st, tail_only=min(T, 5))
        mx.eval(lg)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(R):
        lg, _ = model.step(xt, st, tail_only=min(T, 5))
        a = mx.argmax(lg[0], axis=-1)
        if sync_cpu:
            mx.eval(a)
            int(np.array(a.tolist())[-1])
        else:
            mx.eval(a)
    mx.synchronize()
    return (time.perf_counter() - t0) / R * 1000


def main():
    print(f"полоса памяти (замер): {BW} ГБ/с\n")
    for label, pth, stats in (("черновик 0.1B", DRAFT_PTH, None),
                              ("цель 1.5B", TARGET_PTH, ACT_STATS)):
        model, mb = build(pth, stats)
        roof = mb / 1000 / BW * 1000
        print(f"{label}: {mb:.0f} МБ/ток, потолок по полосе {roof:.2f} мс")
        for T in (1, 5, 9):
            t_sync = timed(model, T, True)
            t_free = timed(model, T, False)
            print(f"   T={T}: {t_sync:6.2f} мс (с sync)   {t_free:6.2f} мс "
                  f"(без sync)   sync стоит {t_sync-t_free:5.2f} мс   "
                  f"эффективность {100*roof/t_free:5.1f}%")
        del model
        gc.collect()
        mx.clear_cache()
        print()


if __name__ == "__main__":
    main()
