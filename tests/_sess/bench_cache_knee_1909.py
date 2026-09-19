"""КОЛЕНО КЭША (19.09): полоса чтения от размера буфера на M4 base.
Вопрос: влезает ли вес следующей GEMV (проекция 2.4 МБ, value 8.4, key 10.7)
в кэш, и насколько быстрее его потом читает ДРУГОЙ кернель -- это премисса
предвыборки весов для цепочки декода.

(а) repeat: один кернель читает буфер R раз (проход p сдвинут на p*N4/3,
    чтобы поток не перечитывал свои же адреса из L1); полоса = S*R/время.
(б) cross: кернель A читает буфер один раз, кернель B (формально зависит от
    A через один элемент его выхода) читает ТОТ ЖЕ буфер. Время B = t(A+B) -
    t(A), сравнивается с t(A) -- холодным чтением. Это и есть сценарий
    «прогрели соседним кернелем -- прочитали из кэша».
Размеры чередуются по раундам (закон 1), медиана. Эталон DRAM -- 104 ГБ/с.
"""
import os, sys, time, subprocess
import numpy as np
import mlx.core as mx

SIZES_MB = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 256]
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "9"))
NTH = 65536
TARGET = 1 << 30          # байт на замер repeat
_k = {}


def kern(n4, passes, dep):
    key = (n4, passes, dep)
    if key in _k:
        return _k[key]
    src = f"""
    uint gid = thread_position_in_grid.x;
    device const uint4* p = (device const uint4*)src;
    uint4 s = uint4({'dep[0]' if dep else '0'});
    for (uint q = 0; q < {passes}; ++q) {{
        uint sh = (q * ({n4}u / 3u + 1u)) % {n4}u;
        for (uint i = gid; i < {n4}u; i += {NTH}u) {{
            uint j = i + sh; if (j >= {n4}u) j -= {n4}u;
            s += p[j];
        }}
    }}
    out[gid] = s.x + s.y + s.z + s.w;
"""
    k = mx.fast.metal_kernel(name=f"knee_{n4}_{passes}_{int(dep)}",
                             input_names=["src", "dep"] if dep else ["src"],
                             output_names=["out"], source=src)
    _k[key] = k
    return k


def run(k, ins):
    return k(inputs=ins, grid=(NTH, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(NTH,)], output_dtypes=[mx.uint32])[0]


def t(fn, reps=5):
    fn(); mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def swap():
    return float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))


sw0 = swap()
bufs = {}
for mb in SIZES_MB:
    n = int(mb * (1 << 20)) // 16 * 4        # uint32 кратно uint4
    bufs[mb] = mx.random.randint(0, 1 << 30, (n,), dtype=mx.uint32)
mx.eval(list(bufs.values()))
empty = mx.zeros((1,), dtype=mx.uint32); mx.eval(empty)

R = {"rep": {m: [] for m in SIZES_MB}, "cold": {m: [] for m in SIZES_MB}, "B": {m: [] for m in SIZES_MB}}
for rd in range(ROUNDS):
    order = SIZES_MB if rd % 2 == 0 else SIZES_MB[::-1]
    for mb in order:
        b = bufs[mb]; n4 = b.size // 4; S = n4 * 16
        passes = max(1, TARGET // S)
        kr = kern(n4, passes, False)
        tr = t(lambda: run(kr, [b]))
        R["rep"][mb].append(S * passes / tr / 1e9)
        k1 = kern(n4, 1, False); k2 = kern(n4, 1, True)
        # A отдельно и A->B; между замерами -- прогон по 256 МБ, выбивающий кэш
        big = bufs[256]; kb = kern(big.size // 4, 1, False)
        def a_only():
            e = run(kb, [big]); return run(k1, [b, ][:1]) + e[0] * 0
        def a_b():
            e = run(kb, [big]); o1 = run(k1, [b]); return run(k2, [b, o1]) + e[0] * 0
        def flush_only():
            return run(kb, [big])
        tf = t(flush_only); ta = t(a_only); tab = t(a_b)
        R["cold"][mb].append(S / max(ta - tf, 1e-9) / 1e9)
        R["B"][mb].append(S / max(tab - ta, 1e-9) / 1e9)

print("размер МБ | repeat ГБ/с (разброс) | холодное A ГБ/с | второе чтение B ГБ/с | B/A")
for mb in SIZES_MB:
    rp = R["rep"][mb]; c = float(np.median(R["cold"][mb])); bb = float(np.median(R["B"][mb]))
    print("%8.1f  | %7.1f (%4.1f%%)       | %7.1f          | %7.1f              | %4.2f" % (
        mb, np.median(rp), 100 * (max(rp) - min(rp)) / np.median(rp), c, bb, bb / c))
print("своп %+.1f МБ" % (swap() - sw0))
