"""05.10: куда уходит память при запуске файла .rwkvq на Metal (вопрос владельца: 7.2B COMPRESSION 4.58 ГБ -> процесс 8.9 ГБ).
Стадии: чтение файла, сборка модели, освобождение сырых массивов, декод, префилл 1024, очистка кеша MLX.
    python mem_probe_0510.py <файл.rwkvq> [T префилла]"""
import gc, os, re, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal import generate as G
F = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
def foot():
    try:
        o = subprocess.run(["/usr/bin/footprint", str(os.getpid())], capture_output=True, text=True).stdout
        m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", o)
        return float(m.group(1)) * {"KB": 1e-3, "MB": 1, "GB": 1024}[m.group(2)]
    except Exception:
        return float("nan")
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))
sw0 = sw()
def stage(name):
    print("%-44s footprint %7.0f МБ | mx активная %7.0f, кеш %7.0f, пик %7.0f МБ | своп %+6.0f МБ" % (
        name, foot(), mx.get_active_memory() / 2**20, mx.get_cache_memory() / 2**20, mx.get_peak_memory() / 2**20, sw() - sw0), flush=True)
    mx.reset_peak_memory()
print("файл %.0f МБ" % (os.path.getsize(F) / 2**20)); stage("старт")
raw = load_raw(F); stage("load_raw (numpy)")
m = QuantRWKV7(raw); mx.eval(m.parameters()) if hasattr(m, "parameters") else None; stage("QuantRWKV7(raw)")
del raw; gc.collect(); stage("del raw + gc")
mx.clear_cache(); stage("mx.clear_cache()")
try:
    from mlx.utils import tree_flatten
    by = {}
    for k, v in tree_flatten(m.parameters()):
        if not hasattr(v, "nbytes"): continue
        p = k.split(".")
        key = re.sub(r"\d+", "N", k)
        key = ".".join(key.split(".")[:4])
        by[key] = by.get(key, 0) + v.nbytes
    tot = sum(by.values())
    print("  параметры модели по группам (МБ), всего %.0f:" % (tot / 2**20))
    for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:14]: print("    %-46s %8.0f" % (k, v / 2**20))
except Exception as e:
    print("  разбор параметров не удался:", type(e).__name__, e)
out = []
for y, _ in G.generate_step(m, [1, 2, 3, 4, 5, 6, 7, 8], 32, pipeline=True, out=out): pass
stage("декод 32 токена")
mx.clear_cache(); stage("mx.clear_cache()")
IDX = mx.array(np.random.default_rng(0).integers(1, 60000, size=(1, T)).astype(np.int32))
lg, st = m.forward_stateful(IDX, m.init_state(1), True); mx.eval(lg); stage("префилл %d токенов (без compile)" % T)
del lg, st; mx.clear_cache(); stage("mx.clear_cache()")
step = mx.compile(m.forward_stateful); lg, st = step(IDX, m.init_state(1), True); mx.eval(lg); stage("префилл %d (mx.compile)" % T)
del lg, st; mx.clear_cache(); stage("mx.clear_cache()")
for t in (256, 64):
    lg, st = m.forward_stateful(IDX[:, :t], m.init_state(1), True); mx.eval(lg); stage("префилл %d токенов" % t)
    del lg, st; mx.clear_cache()
stage("конец")
