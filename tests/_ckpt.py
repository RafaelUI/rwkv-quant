"""Плотный 1.5B-чекпоинт для гейтов на живых тензорах.

Гейты писались на rwkv7-g1h-1.5b (ctx10240); у BlinkDL его больше нет, на его
месте g1i / g1j той же архитектуры (L=24, D=2048). Кернельные и паритетные гейты
от весов не зависят: сверяют два пути на одном и том же тензоре. Порядок:
RWKVQ_CKPT_1P5B, затем g1h, затем g1j — первый существующий файл; если нет
ни одного, возвращается путь g1h (гейт падает с понятным FileNotFoundError).
"""
import os

_CANDIDATES = [
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth",
    "~/Develop/WKV-kvant/rwkv7-g1j-1.5b-20260831-ctx16384.pth",
]


def ckpt_1p5b() -> str:
    env = os.environ.get("RWKVQ_CKPT_1P5B")
    if env:
        return os.path.expanduser(env)
    for c in _CANDIDATES:
        p = os.path.expanduser(c)
        if os.path.exists(p):
            return p
    return os.path.expanduser(_CANDIDATES[0])
