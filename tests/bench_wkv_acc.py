"""WKV-ЯДРО ИНФЕРЕНСА: ДВЕ ОПТИМИЗАЦИИ, КОТОРЫЕ ДО НЕГО НЕ ДОЕХАЛИ.

Рядом в том же пакете живут ДВЕ реализации одной рекуррентности:
`wkv7_checkpoint._get_ckpt_fwd` (обучение) и `wkv7._get_infer_kernel`
(инференс -- то есть наш префилл И наш декод). В первой стоят стейджинг
строк в threadgroup-память ("это стоило ПОЛОВИНЫ времени forward-ядра") и
ACC независимых аккумуляторов вместо одной цепочки из 64 зависимых FMA
("ядро latency-bound"). Во второй -- НИ ОДНОЙ. Закон 20 в чистом виде.

Здесь мерится, сколько каждая из них стоит на форме префилла.

  base      -- как сейчас;
  sh        -- + стейджинг. Порядок суммирования НЕ меняется, поэтому
               требуется РАВЕНСТВО;
  sh+accN   -- + N аккумуляторов. Реассоциация суммы: равенства уже нет,
               ожидается ~1e-7, и это надо не постулировать, а печатать.

ЧЕМ ВРЁТ (закон 29): рабочий набор 24 МБ греется повторами -- абсолюты
завышены; изолированный вызов запрещает перекрытие -- цена запусков
занижена. Арбитр -- сквозной префилл.

    python bench_wkv_acc.py [T] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_metal.kernel.wkv7 import _get_infer_kernel, HEAD_SIZE  # noqa: E402

# ЛОВУШКА (закон 14): в ~/Develop/tests/venv лежит НЕ-editable копия
# rwkv_metal от 25.07, отставшая на несколько правок. Скрипт, импортировавший
# rwkv_metal до rwkv_quant, получил бы её молча. Поэтому путь вставляется
# первым, а исполняемый файл ПЕЧАТАЕТСЯ.
import rwkv_metal.kernel as _km  # noqa: E402
print("исполняется rwkv_metal:", __import__("sys").modules["rwkv_metal"].__file__)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
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


SRC = r"""
    uint dv  = thread_position_in_threadgroup.y;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C; uint hi = bhi % H_C;

    threadgroup float a_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C], k_sh[HEAD_SIZE_C];
    threadgroup float b_sh[HEAD_SIZE_C], r_sh[HEAD_SIZE_C];

    float h_row[HEAD_SIZE_C];
    uint hb = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[hb+dk];

    for (uint t=0; t<CHUNK_C; t++) {
        uint base = ((bi*CHUNK_C+t)*H_C+hi)*HEAD_SIZE_C;
        a_sh[dv]=a[base+dv]; w_sh[dv]=w[base+dv]; k_sh[dv]=k[base+dv];
        b_sh[dv]=b[base+dv]; r_sh[dv]=r[base+dv];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sacc[ACC_C];
        for (uint i=0; i<ACC_C; i++) sacc[i] = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk+=ACC_C)
            for (uint i=0; i<ACC_C; i++) sacc[i] += h_row[dk+i]*a_sh[dk+i];
        float sa = 0.0f;
        for (uint i=0; i<ACC_C; i++) sa += sacc[i];

        float vv = v[base+dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            h_row[dk] = w_sh[dk]*h_row[dk] + vv*k_sh[dk] + sa*b_sh[dk];

        float yacc[ACC_C];
        for (uint i=0; i<ACC_C; i++) yacc[i] = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk+=ACC_C)
            for (uint i=0; i<ACC_C; i++) yacc[i] += h_row[dk+i]*r_sh[dk+i];
        float y = 0.0f;
        for (uint i=0; i<ACC_C; i++) y += yacc[i];
        out[base+dv] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[hb+dk] = h_row[dk];
"""

_cache = {}


def kern_acc(ACC):
    if ACC in _cache:
        return _cache[ACC]
    hdr = (f"\nconstant uint HEAD_SIZE_C = {HEAD_SIZE};"
           f"\nconstant uint CHUNK_C = {T};"
           f"\nconstant uint H_C = {H};"
           f"\nconstant uint ACC_C = {ACC};\n")
    _cache[ACC] = mx.fast.metal_kernel(
        name=f"wkv7_infer_acc{ACC}_{H}_{T}",
        input_names=["r", "w", "k", "v", "a", "b", "h_in"],
        output_names=["out", "h_out"],
        header=hdr, source=SRC,
    )
    return _cache[ACC]


rs = np.random.RandomState(11)


def mk(scale=1.0):
    return mx.array((rs.randn(B, T, H, D) * scale).astype(np.float32))


r = mk(); k = mk(); v = mk(); a = mk() * 0.1; b = mk() * 0.1
w = mx.array((0.9 + 0.09 * rs.rand(B, T, H, D)).astype(np.float32))
h0 = mx.zeros((B, H, D, D), dtype=mx.float32)
mx.eval(r, w, k, v, a, b, h0)

IN = [r, w, k, v, a, b, h0]
OUTS = dict(output_shapes=[(B, T, H, D), (B, H, D, D)],
            output_dtypes=[mx.float32, mx.float32])
base_kern = _get_infer_kernel(H, T)


def call_base():
    return base_kern(inputs=IN, grid=(B * H, D, 1), threadgroup=(1, 1, 1), **OUTS)


def mkcall(ACC):
    def f():
        return kern_acc(ACC)(inputs=IN, grid=(B * H, D, 1),
                             threadgroup=(1, D, 1), **OUTS)
    return f


ACCS = [1, 2, 4, 8, 16]
VARIANTS = [("base", call_base)] + [(f"sh acc={A}", mkcall(A)) for A in ACCS]

bo, bh = [np.asarray(x) for x in call_base()]
print(f"T={T}, форма {tuple(bo.shape)}")
print(f"{'вариант':<12}{'max|dY|':>12}{'rel|dY|':>12}{'max|dH|':>12}")
for name, fn in VARIANTS[1:]:
    o, hh = [np.asarray(x) for x in fn()]
    d1 = np.abs(o - bo).max(); d2 = np.abs(hh - bh).max()
    rel = d1 / max(np.abs(bo).max(), 1e-30)
    tag = " (РАВЕНСТВО)" if d1 == 0 and d2 == 0 else ""
    print(f"{name:<12}{d1:>12.3e}{rel:>12.3e}{d2:>12.3e}{tag}")

for _, fn in VARIANTS:
    mx.eval(fn())

sw0 = swap_mb()
times = {n: [] for n, _ in VARIANTS}
REP = 5
for _ in range(ROUNDS):
    for name, fn in VARIANTS:
        t0 = time.perf_counter()
        for _ in range(REP):
            mx.eval(fn())
        times[name].append((time.perf_counter() - t0) * 1000.0 / REP)
sw1 = swap_mb()

print(f"\nсвоп {sw0:.0f} -> {sw1:.0f} МБ "
      f"({'валиден' if abs(sw1 - sw0) < 1 else 'НЕДЕЙСТВИТЕЛЕН'})")
med = {n: float(np.median(ts)) for n, ts in times.items()}
flops = B * H * D * T * (4 * D) * 2 / 1e9
print(f"\n{'вариант':<12}{'медиана мс':>12}{'мин':>9}{'разброс':>10}"
      f"{'x к base':>10}{'ГФЛОП/с':>10}")
for name, _ in VARIANTS:
    ts = times[name]
    sp = (max(ts) - min(ts)) / np.median(ts) * 100
    print(f"{name:<12}{med[name]:>12.3f}{min(ts):>9.3f}{sp:>9.1f}%"
          f"{med['base']/med[name]:>9.2f}x{flops/(med[name]/1000):>10.0f}")

order = sorted(med.items(), key=lambda kv: kv[1])
print(f"\nпобедитель {order[0][0]}, отрыв от второго ({order[1][0]}) "
      f"{(order[1][1]/order[0][1]-1)*100:.1f}%")
print(f"на 24 слоя: base {med['base']*24:.1f} мс -> "
      f"{order[0][1]*24:.1f} мс (минус {(med['base']-order[0][1])*24:.1f} мс)")
