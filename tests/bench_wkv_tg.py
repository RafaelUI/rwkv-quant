"""ФОРМА THREADGROUP У WKV-ЯДРА: сколько лент SIMD-группы реально работает.

`wkv7_infer` запускается с threadgroup=(1,1,1) при grid=(B*H, D, 1). В Metal
threadgroup из одного потока занимает целую SIMD-группу, поэтому 31 лента из
32 простаивает. Значения от разбиения на threadgroup'ы НЕ зависят вовсе
(тело кернеля адресуется `thread_position_in_grid`), значит это чистая
перекладка запуска -- и выход обязан совпасть БИТ-В-БИТ.

ЧЕМ ЭТОТ ИНСТРУМЕНТ ВРЁТ (закон 29): рабочий набор одного вызова -- шесть
тензоров по 4 МБ, и повторные раунды греют кэш, то есть абсолюты ЗАВЫШЕНЫ
против настоящего префилла, где перед WKV по стеку проехали веса. Плюс
изолированный вызов запрещает перекрытие с соседней работой. Читать отсюда
надо ОТНОШЕНИЕ форм, а решение принимать сквозным замером префилла.

    python bench_wkv_tg.py [T] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_metal.kernel.wkv7 import _get_infer_kernel, HEAD_SIZE  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
B, H, D = 1, 32, HEAD_SIZE


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


rs = np.random.RandomState(11)


def mk(scale=1.0):
    return mx.array((rs.randn(B, T, H, D) * scale).astype(np.float32))


r = mk(); k = mk(); v = mk(); a = mk() * 0.1; b = mk() * 0.1
w = mx.array((0.9 + 0.09 * rs.rand(B, T, H, D)).astype(np.float32))
h0 = mx.zeros((B, H, D, D), dtype=mx.float32)
mx.eval(r, w, k, v, a, b, h0)

kern = _get_infer_kernel(H, T)


def call(tg):
    res = kern(
        inputs=[r, w, k, v, a, b, h0],
        grid=(B * H, D, 1), threadgroup=tg,
        output_shapes=[(B, T, H, D), (B, H, D, D)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return res


VARIANTS = [
    ("(1,1,1) базовая", (1, 1, 1)),
    ("(1,32,1)", (1, 32, 1)),
    ("(1,64,1)", (1, 64, 1)),
    ("(2,32,1)", (2, 32, 1)),
    ("(2,64,1)", (2, 64, 1)),
    ("(4,32,1)", (4, 32, 1)),
    ("(4,64,1)", (4, 64, 1)),
    ("(8,32,1)", (8, 32, 1)),
    ("(32,32,1)", (32, 32, 1)),
]

# --- контроль равенства: битность результата от формы зависеть не должна ---
base_out, base_h = [np.asarray(x) for x in call(VARIANTS[0][1])]
bad = []
for name, tg in VARIANTS[1:]:
    try:
        o, hh = [np.asarray(x) for x in call(tg)]
    except Exception as e:  # неподдержанная форма
        bad.append((name, f"НЕ ЗАПУСТИЛАСЬ: {type(e).__name__}"))
        continue
    d1 = np.abs(o - base_out).max()
    d2 = np.abs(hh - base_h).max()
    if d1 != 0.0 or d2 != 0.0:
        bad.append((name, f"РАСХОЖДЕНИЕ out {d1:.3e} h {d2:.3e}"))
VARIANTS = [(n, tg) for n, tg in VARIANTS
            if n not in {b[0] for b in bad}]
print(f"T={T}, grid=({B*H},{D},1), контроль равенства: "
      f"{'ЗЕЛЁНЫЙ' if not bad else 'ЕСТЬ ПРОБЛЕМЫ'}")
for n, msg in bad:
    print(f"   {n}: {msg}")

# прогрев
for _, tg in VARIANTS:
    mx.eval(call(tg))

sw0 = swap_mb()
times = {n: [] for n, _ in VARIANTS}
REP = 5
for rnd in range(ROUNDS):
    for name, tg in VARIANTS:          # чередование внутри раунда (закон 24)
        t0 = time.perf_counter()
        for _ in range(REP):
            mx.eval(call(tg))
        times[name].append((time.perf_counter() - t0) * 1000.0 / REP)
sw1 = swap_mb()

print(f"\nсвоп: {sw0:.0f} -> {sw1:.0f} МБ "
      f"({'валиден' if abs(sw1 - sw0) < 1 else 'НЕДЕЙСТВИТЕЛЕН'})")
print(f"\n{'форма':<18}{'медиана мс':>12}{'разброс':>10}{'x к базовой':>14}")
med = {n: float(np.median(ts)) for n, ts in times.items()}
base = med[VARIANTS[0][0]]
for name, _ in VARIANTS:
    ts = times[name]
    spread = (max(ts) - min(ts)) / np.median(ts) * 100
    print(f"{name:<18}{med[name]:>12.3f}{spread:>9.1f}%{base / med[name]:>13.2f}x")

order = sorted(med.items(), key=lambda kv: kv[1])
if len(order) > 1:
    print(f"\nпобедитель {order[0][0]} отрывается от второго "
          f"({order[1][0]}) на {(order[1][1] / order[0][1] - 1) * 100:.1f}%")

# арифметика: сколько это в ФЛОПах и байтах
flops = B * H * D * T * (4 * D) * 2  # sa, h-update(2 mul-add по 64), y
traffic = (6 * B * T * H * D + B * T * H * D) * 4 / 1e6
print(f"\nна вызов: {flops/1e9:.2f} ГФЛОП, {traffic:.1f} МБ трафика минимум")
for name, _ in order[:3]:
    print(f"   {name:<18}{flops/1e9/(med[name]/1000):>8.0f} ГФЛОП/с  "
          f"{traffic/(med[name]/1000)/1000:>7.1f} ГБ/с")
