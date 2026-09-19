"""ЦЕПОЧКА ЗАВИСИМЫХ ЧТЕНИЙ (19.09) -- модель цепочки декода.
64 кернеля подряд в одном eval, каждый формально зависит от предыдущего
(берёт один элемент его выхода) и читает буфер размера S целиком:
  warm -- все 64 читают ОДИН буфер (со 2-го -- из кэша, если влезает);
  cold -- буферы по кругу из набора суммой >= 256 МБ (каждое чтение из DRAM).
Отношение cold к 104 ГБ/с -- потеря полосы на зависимой цепочке данного
размера (то, что теряет декод); warm против cold -- что даёт кэш, если вес
заранее в нём. Размеры чередуются по раундам, медиана.
"""
import os, time, subprocess
import numpy as np
import mlx.core as mx

SIZES_MB = [float(x) for x in os.environ.get("RWKVQ_SIZES", "1,2,2.4,3,4,6,7.1,8.4,10.7,12,16,24,32").split(",")]
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "9"))
N = 64
NTH = 65536
_k = {}


def kern(n4):
    if n4 in _k:
        return _k[n4]
    src = f"""
    uint gid = thread_position_in_grid.x;
    device const uint4* p = (device const uint4*)src;
    uint4 s = uint4(dep[0] & 0u);
    for (uint i = gid; i < {n4}u; i += {NTH}u) s += p[i];
    out[gid] = s.x + s.y + s.z + s.w;
"""
    k = mx.fast.metal_kernel(name=f"chain_{n4}", input_names=["src", "dep"],
                             output_names=["out"], source=src)
    _k[n4] = k
    return k


def chain(k, bufs):
    o = mx.zeros((1,), dtype=mx.uint32)
    for i in range(N):
        o = k(inputs=[bufs[i % len(bufs)], o], grid=(NTH, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(NTH,)], output_dtypes=[mx.uint32])[0]
    return o


def t(fn, reps=5):
    mx.eval(fn()); mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def swap():
    return float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))


sw0 = swap()
SETS = {}
for mb in SIZES_MB:
    n = int(mb * (1 << 20)) // 64 * 16
    m = max(2, int(np.ceil(256 / mb)))
    SETS[mb] = [mx.random.randint(0, 1 << 30, (n,), dtype=mx.uint32) for _ in range(m)]
    mx.eval(SETS[mb])
W = {m: [] for m in SIZES_MB}; C = {m: [] for m in SIZES_MB}
for rd in range(ROUNDS):
    for mb in (SIZES_MB if rd % 2 == 0 else SIZES_MB[::-1]):
        bs = SETS[mb]; k = kern(bs[0].size // 4); S = bs[0].size * 4
        tw = t(lambda: chain(k, bs[:1])); tc = t(lambda: chain(k, bs))
        W[mb].append((tw / N * 1e6, S / (tw / N) / 1e9)); C[mb].append((tc / N * 1e6, S / (tc / N) / 1e9))
print("S МБ  | cold: мкс/кернель  ГБ/с  %%от104 (разброс) | warm: мкс/кернель  ГБ/с | warm/cold")
for mb in SIZES_MB:
    cu = np.median([x[0] for x in C[mb]]); cg = np.median([x[1] for x in C[mb]])
    wu = np.median([x[0] for x in W[mb]]); wg = np.median([x[1] for x in W[mb]])
    sp = 100 * (max(x[1] for x in C[mb]) - min(x[1] for x in C[mb])) / cg
    print("%5.1f | %8.1f %7.1f %5.0f%% (%4.1f%%)            | %8.1f %7.1f | %5.2f" % (mb, cu, cg, 100 * cg / 104, sp, wu, wg, cu / wu))
print("своп %+.1f МБ" % (swap() - sw0))
