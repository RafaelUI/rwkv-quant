"""ПАМЯТЬ БАТЧЕВОГО ПРЕФИЛЛА T=4096: ОДИН БАТЧ НА ПРОЦЕСС.

Счётчики памяти MLX здесь НЕ используются: они врут (решение владельца
15.09). Мерится тем, что видит система: пик берётся из /usr/bin/time -l
снаружи (peak memory footprint -- та метрика, что работает для unified
memory, закон 22), а изнутри печатаются своп и vm_stat до и после,
ДЕЛЬТОЙ (закон 11).

Один конфиг на процесс (закон 2): иначе пик принадлежал бы самому
большому батчу прогона, а не своему. Первый вызов на форму
выбрасывается -- в нём трассировка mx.compile и первое касание страниц
(закон 26), но печатается тоже: его цена сама по себе интересна.

Swapins/Swapouts из vm_stat -- счётчики накопительные, поэтому их дельта
отвечает на вопрос «свопило ли ВООБЩЕ», чего vm.swapusage не показывает:
занятый своп может не двигаться при активном обмене.
"""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

KV = "/Users/s/Develop/WKV-kvant/"
COMP = KV + os.environ.get("RWKVQ_COMP", "compression_v2_cand.rwkvq")
T = int(os.environ.get("RWKVQ_T", "4096"))
B = int(os.environ.get("RWKVQ_B", "1"))
PAGE = 16384.0
KEYS = ("Pages free", "Pages active", "Pages wired down", "Swapins", "Swapouts")


def vmstat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {}
    for ln in out.splitlines():
        for k in KEYS:
            if ln.startswith(k + ":"):
                d[k] = int(ln.split(":")[1].strip().rstrip("."))
    return d


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def report(tag, a, b, sa, sb):
    print("  %s: своп %+.1f МБ; активных %+.0f МБ, wired %+.0f МБ, свободных %+.0f МБ; swapin %+d, swapout %+d" % (
        tag, sb - sa,
        (b["Pages active"] - a["Pages active"]) * PAGE / 1e6,
        (b["Pages wired down"] - a["Pages wired down"]) * PAGE / 1e6,
        (b["Pages free"] - a["Pages free"]) * PAGE / 1e6,
        b["Swapins"] - a["Swapins"], b["Swapouts"] - a["Swapouts"]), flush=True)


print("B=%d T=%d файл %s" % (B, T, os.path.basename(COMP)), flush=True)
v0, s0 = vmstat(), swap_used()
m = qm.QuantRWKV7(load_raw(COMP))
rng = np.random.default_rng(1000 + B)
idx = mx.array(rng.integers(1, 60000, size=(B, T)).astype(np.int32))
v1, s1 = vmstat(), swap_used()
report("загрузка модели", v0, v1, s0, s1)

step = m.step
st = m.init_state(B)
t0 = time.perf_counter()
lg, _ = step(idx, st, True)
mx.eval(lg)
mx.synchronize()
warm = (time.perf_counter() - t0) * 1e3
v2, s2 = vmstat(), swap_used()
report("первый вызов (с трассировкой)", v1, v2, s1, s2)
print("  первый вызов %.1f мс (выбрасывается), форма логитов %s" % (
    warm, str(lg.shape)), flush=True)
del lg

st = m.init_state(B)
t0 = time.perf_counter()
lg, _ = step(idx, st, True)
mx.eval(lg)
mx.synchronize()
dt = (time.perf_counter() - t0) * 1e3
v3, s3 = vmstat(), swap_used()
report("замерный вызов", v2, v3, s2, s3)
print("  ЗАМЕР B=%d: %.1f мс = %.1f ток/с всего, %.1f ток/с на поток" % (
    B, dt, 1000.0 * B * T / dt, 1000.0 * T / dt), flush=True)
report("весь процесс", v0, v3, s0, s3)
print("ГОТОВО B=%d" % B, flush=True)
