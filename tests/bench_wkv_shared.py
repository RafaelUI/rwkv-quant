"""WKV НА ПРЕФИЛЛЕ: ГДЕ УХОДИТ ВРЕМЯ. Три реализации одной формулы.

Наблюдение, с которого всё началось. В `wkv7_infer` каждый из 64 потоков
одной пары (b, h) в КАЖДОМ из внутренних циклов читает ВСЮ строку
a/w/k/b/r из глобальной памяти. То есть одна и та же строка читается 64
раза: 2048 потоков x 512 шагов x 5 строк x 256 байт = 1.34 ГБ загрузок на
вызов при 29.4 МБ полезного трафика -- в 46 раз больше необходимого.
Форма threadgroup при этом ни при чём (замерено: (1,1,1) и (1,64,1) дают
одно и то же до 0.1%).

Варианты:
  base    -- как сейчас;
  shared  -- строки складываются в threadgroup-память один раз на шаг,
             арифметика и ПОРЯДОК СУММИРОВАНИЯ те же, значит выход обязан
             быть БИТ-В-БИТ равен базовому;
  shared2 -- то же, но threadgroup накрывает две головы (128 потоков):
             больше работы на диспатч при том же трафике.

ЧЕМ ВРЁТ (закон 29): рабочий набор вызова 24 МБ, повторные раунды его
греют -- абсолюты ЗАВЫШЕНЫ против настоящего префилла; изолированный
вызов запрещает перекрытие -- цена запусков ЗАНИЖЕНА. Арбитр -- сквозной
префилл.

    python bench_wkv_shared.py [T] [раундов]
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


def _hdr(HPG):
    return f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {T};
constant uint H_C         = {H};
constant uint HPG_C       = {HPG};
"""


# HPG -- сколько пар (b,h) обслуживает одна threadgroup.
SHARED_SRC = r"""
    uint dv   = thread_position_in_threadgroup.y;
    uint lh   = thread_position_in_threadgroup.x;        // 0..HPG_C-1
    uint bhi  = threadgroup_position_in_grid.x * HPG_C + lh;
    uint bi   = bhi / H_C; uint hi = bhi % H_C;

    threadgroup float sh[HPG_C][6][HEAD_SIZE_C];
    threadgroup float *a_sh = sh[lh][0];
    threadgroup float *w_sh = sh[lh][1];
    threadgroup float *k_sh = sh[lh][2];
    threadgroup float *b_sh = sh[lh][3];
    threadgroup float *r_sh = sh[lh][4];
    threadgroup float *v_sh = sh[lh][5];

    float h_row[HEAD_SIZE_C];
    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];

    for (uint t=0; t<CHUNK_C; t++) {
        uint base = ((bi*CHUNK_C+t)*H_C+hi)*HEAD_SIZE_C;
        a_sh[dv]=a[base+dv]; w_sh[dv]=w[base+dv]; k_sh[dv]=k[base+dv];
        b_sh[dv]=b[base+dv]; r_sh[dv]=r[base+dv]; v_sh[dv]=v[base+dv];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sa = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a_sh[dk];
        float v_dv = v_sh[dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            h_row[dk] = w_sh[dk]*h_row[dk] + v_dv*k_sh[dk] + sa*b_sh[dk];
        float y = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r_sh[dk];
        out[base+dv] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
"""

_cache = {}


def shared_kernel(HPG):
    if HPG in _cache:
        return _cache[HPG]
    _cache[HPG] = mx.fast.metal_kernel(
        name=f"wkv7_infer_sh_{H}_{T}_{HPG}",
        input_names=["r", "w", "k", "v", "a", "b", "h_in"],
        output_names=["out", "h_out"],
        header=_hdr(HPG), source=SHARED_SRC,
    )
    return _cache[HPG]


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


def call_shared(HPG):
    return shared_kernel(HPG)(
        inputs=IN, grid=(B * H, D, 1), threadgroup=(HPG, D, 1), **OUTS)


VARIANTS = [("base", call_base)]
for hpg in (1, 2, 4):
    VARIANTS.append((f"shared HPG={hpg}", (lambda p: lambda: call_shared(p))(hpg)))

# --- контроль: РАВЕНСТВО, а не порог. Порядок суммирования не менялся ---
bo, bh = [np.asarray(x) for x in call_base()]
bad = []
for name, fn in VARIANTS[1:]:
    try:
        o, hh = [np.asarray(x) for x in fn()]
    except Exception as e:
        bad.append((name, f"НЕ СОБРАЛСЯ: {type(e).__name__}: {str(e)[:200]}"))
        continue
    d1 = np.abs(o - bo).max(); d2 = np.abs(hh - bh).max()
    if d1 or d2:
        bad.append((name, f"РАСХОЖДЕНИЕ out {d1:.3e} h {d2:.3e}"))
print(f"T={T}: контроль равенства "
      f"{'ЗЕЛЁНЫЙ' if not bad else 'КРАСНЫЙ'}")
for n, m in bad:
    print(f"   {n}: {m}")
VARIANTS = [(n, f) for n, f in VARIANTS if n not in {x[0] for x in bad}]

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
useful = (7 * B * T * H * D) * 4 / 1e6
redundant = (2048 * T * 5 * D * 4) / 1e6
print(f"\n{'вариант':<16}{'мс':>9}{'разброс':>10}{'x к base':>10}"
      f"{'ГФЛОП/с':>10}{'загрузок ГБ/с':>15}")
for name, _ in VARIANTS:
    ts = times[name]
    sp = (max(ts) - min(ts)) / np.median(ts) * 100
    ld = useful if name != "base" else redundant
    print(f"{name:<16}{med[name]:>9.3f}{sp:>9.1f}%{med['base']/med[name]:>9.2f}x"
          f"{flops/(med[name]/1000):>10.0f}{ld/(med[name]/1000)/1000:>14.1f}")
print(f"\nполезный трафик {useful:.1f} МБ, у base фактических загрузок "
      f"{redundant:.0f} МБ ({redundant/useful:.0f}x)")
print(f"на 24 слоя: base {med['base']*24:.1f} мс, "
      f"лучший {min(med.values())*24:.1f} мс")
