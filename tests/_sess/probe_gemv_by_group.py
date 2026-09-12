"""Полоса ПО ГРУППАМ ФОРМ внутри GEMV-фазы, чередованием в одном процессе.

09.09: GEMV-фаза целиком идёт на 80.2 ГБ/с при потолке 104, но голова внутри
неё -- на 89.6, то есть потеря сидит в теле. Формы там разные: proj 2048->2048
при 4 битах, cmix.key 2048->8192 при 5, cmix.value 8192->2048 при 4. У них
разная длина редукции и разное число выходов, и мешать их в одно число нельзя.
"""
import gc
import os
import subprocess
import sys

sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")

import mlx.core as mx
import numpy as np
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]
ROUNDS, REPS, WARM = 9, 7, 3


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    return float(num) * (1024 if unit == "G" else 1)


def bench(fn):
    for _ in range(WARM):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(REPS):
        import time
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    m = QuantRWKV7(load_raw(PATH))
    gc.collect()
    mx.clear_cache()
    D = int(m.emb_weight.shape[1])
    x = mx.array(np.random.randn(1, D).astype(np.float32))
    xf = mx.array(np.random.randn(1, 4 * D).astype(np.float32))
    mx.eval(x, xf)

    def bytes_of(l):
        return sum(v.nbytes for v in vars(l).values() if isinstance(v, mx.array))

    tb = {"proj r/k/v/o": 0, "cmix.key": 0, "cmix.value": 0, "head": bytes_of(m.head)}
    for b in m.blocks:
        for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj):
            tb["proj r/k/v/o"] += bytes_of(l)
        tb["cmix.key"] += bytes_of(b.cmix.key)
        tb["cmix.value"] += bytes_of(b.cmix.value)

    def f_proj():
        return [l(x) for b in m.blocks
                for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj)]

    def f_key():
        return [b.cmix.key(x) for b in m.blocks]

    def f_val():
        return [b.cmix.value(xf) for b in m.blocks]

    def f_head():
        return [m.head(x)]

    cases = {"proj r/k/v/o": f_proj, "cmix.key": f_key,
             "cmix.value": f_val, "head": f_head}
    sw0 = swap_mb()
    acc = {k: [] for k in cases}
    for _ in range(ROUNDS):
        for k, fn in cases.items():
            acc[k].append(bench(fn))
    sw1 = swap_mb()
    print("файл: %s" % os.path.basename(PATH), flush=True)
    print("своп %.0f -> %.0f МБ%s" % (sw0, sw1,
          "  *** РОС, ЗАМЕР НЕДЕЙСТВИТЕЛЕН (закон 11) ***" if sw1 > sw0 + 0.5 else "  (не рос)"), flush=True)
    print("", flush=True)
    print("%-14s %8s %8s %9s %8s %8s" % ("группа", "мс", "разброс", "МБ", "ГБ/с", "%% от 104"), flush=True)
    tot_ms = tot_mb = 0.0
    for k in cases:
        med = float(np.median(acc[k]))
        sp = 100 * (max(acc[k]) - min(acc[k])) / med
        mb = tb[k] / 1e6
        bw = mb / med
        tot_ms += med
        tot_mb += mb
        print("%-14s %8.3f %7.1f%% %9.1f %8.1f %7.0f%%" % (k, med, sp, mb, bw, 100 * bw / 104), flush=True)
    print("%-14s %8.3f %8s %9.1f %8.1f %7.0f%%" % ("ИТОГО", tot_ms, "", tot_mb, tot_mb / tot_ms, 100 * (tot_mb / tot_ms) / 104), flush=True)


if __name__ == "__main__":
    main()
