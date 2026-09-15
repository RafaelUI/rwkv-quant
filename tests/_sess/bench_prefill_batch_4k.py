"""СКОРОСТЬ батчевого префилла T=4096, чередованием плеч в ОДНОМ процессе.

Память меряет ДРУГОЙ прибор -- bench_prefill_batch_mem.py, по одному
батчу на процесс, пиком из /usr/bin/time -l. Счётчики памяти MLX не
используются нигде: они врут (решение владельца 15.09). Здесь из памяти
печатается только своп и swapout ДЕЛЬТОЙ -- как условие годности замера
(закон 11), а не как измеряемая величина.

ОБЛАСТЬ ДЕЙСТВИЯ: только батчи, влезающие в ОЗУ. Пред-замер 15.09 дал пик
5.11 ГиБ при B=1, 7.78 при B=2, 11.73 при B=4 (и B=4 уже свопил на
+2054 МБ), а B=8 утащил в своп 7.4 ГБ и был снят. Поэтому чередуются
B=1 и B=2; B=4 и B=8 в сравнение скорости НЕ входят -- своп во время
замера делает замер недействительным, а не медленным.

T=4096 идёт ОДНИМ вызовом forward_stateful, без деления на куски.
last_only=True -- голова только на последней позиции, как в настоящем
префилле генерации.

Первый вызов на форму выбрасывается: в нём трассировка mx.compile
(записано 110-406 мс на форму) и первое касание страниц (закон 26).
Плечо B=1 печатается пораундово -- на прогоне такой длины виден
безвентиляторный дрейф (закон 25), и отношение защищено только
чередованием.
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
BS = [int(x) for x in os.environ.get("RWKVQ_BS", "1,2").split(",")]
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "3"))


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.split()
    return float(out[5].rstrip("M"))


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if ln.startswith("Swapouts:"):
            return int(ln.split(":")[1].strip().rstrip("."))
    return -1


def make_pf(m, B):
    rng = np.random.default_rng(1000 + B)
    idx = mx.array(rng.integers(1, 60000, size=(B, T)).astype(np.int32))
    step = m.step

    def pf():
        st = m.init_state(B)
        lg, _ = step(idx, st, True)
        return lg
    return pf


sw0, so0 = swap_used(), swapouts()
print("своп до: %.1f МБ; T=%d, батчи %s, раундов %d" % (sw0, T, str(BS), ROUNDS), flush=True)
m = qm.QuantRWKV7(load_raw(COMP))
print("файл %s пресет %s fast_ln %s" % (
    os.path.basename(COMP), str(m.preset), str(m.fast_ln)), flush=True)
PF = [(B, make_pf(m, B)) for B in BS]

for B, fn in PF:
    t0 = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    print("  прогрев B=%d (с трассировкой): %.1f мс -- выброшен" % (
        B, (time.perf_counter() - t0) * 1e3), flush=True)

res = dict((B, []) for B in BS)
for r in range(ROUNDS):
    for B, fn in PF:
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        res[B].append((time.perf_counter() - t0) * 1e3)
    print("  раунд %d: %s" % (r + 1, ", ".join(
        "B=%d %.0f мс" % (B, res[B][-1]) for B in BS)), flush=True)

sw1, so1 = swap_used(), swapouts()
print("своп за прогон: %+.1f МБ, swapout %+d страниц -- %s" % (
    sw1 - sw0, so1 - so0,
    "замер годен" if (so1 - so0) < 1000 else "СВОП ДВИНУЛСЯ, ЗАМЕР НЕДЕЙСТВИТЕЛЕН"), flush=True)

base = float(np.median(res[BS[0]]))
print("--- ТАБЛИЦА: T=%d одним вызовом ---" % T, flush=True)
print("  B  токенов  медиана мс  разброс  ток/с всего  ток/с на поток  x к B=1", flush=True)
for B in BS:
    v = res[B]
    md = float(np.median(v))
    print("  %-2d %7d  %10.1f  %5.1f%%  %11.1f  %14.1f  %6.2fx" % (
        B, B * T, md, 100.0 * (max(v) - min(v)) / md,
        1000.0 * B * T / md, 1000.0 * T / md, (base * B) / md), flush=True)
v = res[BS[0]]
print("  дрейф на плече B=%d: раунд 1 %.0f мс -> раунд %d %.0f мс = %+.1f%%" % (
    BS[0], v[0], len(v), v[-1], 100.0 * (v[-1] - v[0]) / v[0]), flush=True)
print("ГОТОВО", flush=True)
