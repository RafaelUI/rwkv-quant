"""ПРЕДВЫБОРКА НА МОДЕЛИ ЦЕПОЧКИ (19.09). Цепочка зависимых чтений с
размерами слоя 1.5B (rkv 9.1, o 3.0, key 10.7, value 8.4 МБ; 24 слоя, разные
буферы -- как веса, каждый читается раз за токен), плюс голова 94.4 МБ.
  base -- как декод: звено i+1 ждёт звено i;
  pf   -- параллельно звену i кернель pf_i читает буфер звена i+1 (от цепочки
          не зависит), а звено i+1 формально ждёт и pf_i: к его старту вес уже
          прочитан, возможно в кэш. Байт из DRAM не меньше -- заполняются
          пузыри разгона/хвоста;
  pf_half -- pf читает только первую половину следующего буфера (если кэш не
          вмещает целый вес, может быть лучше).
Чередование, медиана, ГБ/с = байты цепочки / время.
"""
import os, time, subprocess
import numpy as np
import mlx.core as mx

NTH = 65536
ROUNDS = int(os.environ.get("RWKVQ_ROUNDS", "9"))
LAYER = [9.1, 3.0, 10.7, 8.4]
NL = int(os.environ.get("RWKVQ_NL", "24"))
_k = {}


def kern(n4, frac=1.0):
    key = (n4, frac)
    if key in _k:
        return _k[key]
    lim = int(n4 * frac)
    src = f"""
    uint gid = thread_position_in_grid.x;
    device const uint4* p = (device const uint4*)src;
    uint4 s = uint4(dep[0] & 0u) + uint4(dep2[0] & 0u);
    for (uint i = gid; i < {lim}u; i += {NTH}u) s += p[i];
    out[gid] = s.x + s.y + s.z + s.w;
"""
    k = mx.fast.metal_kernel(name=f"pf_{n4}_{lim}", input_names=["src", "dep", "dep2"],
                             output_names=["out"], source=src)
    _k[key] = k
    return k


def call(k, b, d1, d2):
    return k(inputs=[b, d1, d2], grid=(NTH, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(NTH,)], output_dtypes=[mx.uint32])[0]


sizes = LAYER * NL + [94.4]
bufs = [mx.random.randint(0, 1 << 30, (int(mb * 1e6) // 64 * 16,), dtype=mx.uint32) for mb in sizes]
mx.eval(bufs)
BYTES = sum(b.size * 4 for b in bufs)
z = mx.zeros((1,), dtype=mx.uint32); mx.eval(z)


def run(mode):
    o = z
    pf = z
    for i, b in enumerate(bufs):
        o = call(kern(b.size // 4), b, o, pf)
        if mode != "base" and i + 1 < len(bufs):
            nb = bufs[i + 1]
            frac = 0.5 if mode == "pf_half" else 1.0
            pf = call(kern(nb.size // 4, frac), nb, z, z)   # не зависит от цепочки
        else:
            pf = z
    return o


def t(mode, reps=5):
    mx.eval(run(mode)); mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(run(mode)); mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def swap():
    return float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))


sw0 = swap()
modes = ["base", "pf", "pf_half"]
V = {m: [] for m in modes}
for rd in range(ROUNDS):
    for m in (modes if rd % 2 == 0 else modes[::-1]):
        V[m].append(t(m) * 1e3)
print("цепочка: %d звеньев, %.1f МБ" % (len(bufs), BYTES / 1e6))
for m in modes:
    med = np.median(V[m])
    print("  %-8s %7.3f мс  %6.1f ГБ/с = %3.0f%% от 104  разброс %4.1f%%" % (
        m, med, BYTES / 1e6 / med, 100 * BYTES / 1e6 / med / 104, 100 * (max(V[m]) - min(V[m])) / med))
print("своп %+.1f МБ" % (swap() - sw0))
