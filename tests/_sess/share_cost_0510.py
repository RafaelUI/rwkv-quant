"""05.10: цена RWKVQ_RKV_SHARE на декоде. В одном процессе чередуются два скомпилированных шага: слитый (FUSE=True, рабочий путь)
и неслитый (FUSE=False: r/k/v тремя отдельными вызовами -- при SHARE они читают СРЕЗЫ буфера фьюза на каждый токен).
Запускать дважды: RWKVQ_RKV_SHARE=0 и =1; сравнивать отношение неслитый / слитый и абсолюты.
    python share_cost_0510.py <файл.rwkvq>"""
import os, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal import quant_model as Q
m = Q.QuantRWKV7(load_raw(sys.argv[1]))
nf = sum(1 for b in m.blocks if getattr(b.tmix, "_rkv_fused", None) is not None)
rng = np.random.default_rng(0)
tok = lambda: mx.array(rng.integers(1, 60000, size=(1, 1)).astype(np.int32))
steps = {}
for name, flag in (("слитый", True), ("неслитый", False)):
    Q.FUSE = flag
    f = mx.compile(m.forward_stateful); s = m.init_state(1)
    for _ in range(6):
        lg, s = f(tok(), s); mx.eval(lg, s)
    steps[name] = [f, s, flag]
nf = sum(1 for b in m.blocks if getattr(b.tmix, "_rkv_fused", None) is not None)
T = {k: [] for k in steps}
order = ["слитый", "неслитый", "неслитый", "слитый"]
for r in range(10):
    for name in (order if r % 2 == 0 else order[::-1]):
        f, s, flag = steps[name]; Q.FUSE = flag; ts = []
        for _ in range(8):
            x = tok(); t0 = time.perf_counter(); lg, s = f(x, s); mx.eval(lg, s); ts.append(time.perf_counter() - t0)
        steps[name][1] = s; T[name].append(1e3 * float(np.median(ts)))
a, b = np.array(T["слитый"]), np.array(T["неслитый"])
print("%s | SHARE=%s | слоёв со слитым r/k/v %d из %d | слитый %.2f мс/ток, неслитый %.2f мс/ток, неслитый / слитый x%.3f | mx активная %.0f МиБ" % (
    os.path.basename(sys.argv[1]), os.environ.get("RWKVQ_RKV_SHARE", "0"), nf, len(m.blocks), np.median(a), np.median(b), np.median(b) / np.median(a), mx.get_active_memory() / 2**20), flush=True)
