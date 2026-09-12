"""Конфигурации четырёх чекпоинтов под развёртку по размерам.

Нужны n_layer (ось компаундинга), n_embd и head_size (геометрия ядра), число
параметров (оценка размера .rwkvq и воркспейса квантования). torch.load ТОЛЬКО
с mmap=True: без него плечо съедает 10.3 ГБ и уводит своп.
"""
import os
import sys

import torch

PATHS = [
    "/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth",
    "/Users/s/rwkv7-g1d-0.4b-ctx8192.pth",
    "/Users/s/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth",
    "/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth",
]

print("%-30s %7s %6s %5s %5s %9s %8s" % ("файл", "слоёв", "D", "S", "H", "параметров", "на диске"), flush=True)
for p in PATHS:
    sd = torch.load(p, map_location="cpu", mmap=True)
    nl = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    emb = sd["emb.weight"]
    D = int(emb.shape[1])
    rk = next(v for k, v in sd.items() if k.endswith("r_k"))
    S = int(rk.shape[-1])
    n = sum(v.numel() for v in sd.values())
    print("%-30s %7d %6d %5d %5d %9.3fB %7.1fГБ"
          % (os.path.basename(p)[:30], nl, D, S, D // S, n / 1e9,
             os.path.getsize(p) / 1e9), flush=True)
    del sd
