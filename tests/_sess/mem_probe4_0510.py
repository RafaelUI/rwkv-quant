"""05.10: память и цена правок памяти (полосы хост -> MLX, ленивая склейка LoRA, RWKVQ_RKV_SHARE): footprint и Malloc Large
после сборки и после первого декода; время префилла T=16 / 256 и декода через скомпилированный model.step.
    [RWKVQ_RKV_SHARE=1] python mem_probe4_0510.py <файл.rwkvq>"""
import gc, os, re, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
def fp():
    o = subprocess.run(["/usr/bin/footprint", str(os.getpid())], capture_output=True, text=True).stdout
    g = lambda m: float(m.group(1)) * {"KB": 1e-3, "MB": 1, "GB": 1024}[m.group(2)] if m else 0.0
    return g(re.search(r"phys_footprint: ([\d.]+) (KB|MB|GB)", o)), g(re.search(r"([\d.]+) (KB|MB|GB)\s+\S+ \S+\s+\S+ \S+\s+\d+\s+Malloc Large", o))
def st(tag): print("%-34s footprint %6.0f | Malloc Large %5.0f | mx активная %6.0f, кеш %5.0f МБ" % (tag, *fp(), mx.get_active_memory() / 2**20, mx.get_cache_memory() / 2**20), flush=True)
print("RKV_SHARE =", os.environ.get("RWKVQ_RKV_SHARE", "0"))
raw = load_raw(sys.argv[1]); m = QuantRWKV7(raw); del raw; gc.collect(); st("после сборки")
rng = np.random.default_rng(0)
ids = lambda T: mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
lg, s = m.step(ids(8), m.init_state(1)); mx.eval(lg, s)
for _ in range(8):
    lg, s = m.step(ids(1), s); mx.eval(lg, s)
mx.clear_cache(); st("после префилла 8 + декода 8")
ts = []
for _ in range(24):
    x = ids(1); t0 = time.perf_counter(); lg, s = m.step(x, s); mx.eval(lg, s); ts.append(time.perf_counter() - t0)
print("декод: %.2f мс/ток (медиана 24)" % (1e3 * float(np.median(ts))))
for T, n in ((16, 12), (256, 4)):
    x = ids(T); lg, s2 = m.step(x, m.init_state(1)); mx.eval(lg, s2); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); lg, s2 = m.step(x, m.init_state(1)); mx.eval(lg, s2); ts.append(time.perf_counter() - t0)
    print("префилл T=%d: %.1f мс (медиана %d), %.0f т/с" % (T, 1e3 * float(np.median(ts)), n, T / float(np.median(ts))))
mx.clear_cache(); st("конец")
if os.environ.get("MP_DETAIL"):
    def detail(tag):
        o = subprocess.run(["/usr/bin/footprint", str(os.getpid())], capture_output=True, text=True).stdout
        print("--", tag)
        for l in o.split("\n"):
            if re.search(r"^\s*[\d.]+ (MB|GB)", l): print("   ", l.rstrip()[:100])
    detail("сразу"); time.sleep(5); detail("через 5 с"); del lg, s, s2, x; gc.collect(); mx.clear_cache(); time.sleep(2); detail("после del выходов + clear_cache")
    print("mx активная %.0f кеш %.0f пик %.0f" % (mx.get_active_memory() / 2**20, mx.get_cache_memory() / 2**20, mx.get_peak_memory() / 2**20))
