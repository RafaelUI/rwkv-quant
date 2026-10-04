import gc, os, re, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal import generate as G
def foot():
    o = subprocess.run(["/usr/bin/footprint", str(os.getpid())], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", o); return float(m.group(1)) * {"KB": 1e-3, "MB": 1, "GB": 1024}[m.group(2)]
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))
sw0 = sw()
def stage(n): print("%-40s footprint %6.0f | активная %6.0f кеш %5.0f пик %6.0f МБ | своп %+5.0f" % (n, foot(), mx.get_active_memory()/2**20, mx.get_cache_memory()/2**20, mx.get_peak_memory()/2**20, sw()-sw0), flush=True); mx.reset_peak_memory()
mx.set_cache_limit(64 * 2**20)
m = QuantRWKV7(load_raw(sys.argv[1])); gc.collect(); stage("загрузка, кеш MLX ограничен 64 МБ")
IDX = mx.array(np.random.default_rng(0).integers(1, 60000, size=(1, 1024)).astype(np.int32))
for C in (1024, 256, 128):
    st = m.init_state(1); mx.synchronize(); t0 = time.perf_counter(); pk = 0
    for i in range(0, 1024, C):
        lg, st = m.forward_stateful(IDX[:, i:i + C], st, True); mx.eval(lg, st); pk = max(pk, foot())
    dt = time.perf_counter() - t0
    stage("префилл 1024 кусками по %d: %.0f т/с, пик footprint %.0f" % (C, 1024 / dt, pk))
out = []
t0 = time.perf_counter()
for y, _ in G.generate_step(m, [int(x) for x in np.array(IDX[0, :8])], 64, pipeline=True, out=out): pass
stage("декод 64 токена: %.1f т/с" % (64 / (time.perf_counter() - t0)))
