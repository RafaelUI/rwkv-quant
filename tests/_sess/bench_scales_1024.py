"""pp1024 и tg1024 одного .rwkvq в одном процессе (17.09): таблица скорости
по масштабам против llama.cpp -p 1024 -n 1024.

Префилл: один вызов `forward_stateful(last_only=True)` под `mx.compile` на
T=1024 -- ровно то, что llama.cpp считает при обработке промпта (логиты
только последней позиции). Декод: 1024 шага по одному токену, состояние
живое, argmax на GPU, синк каждый токен (так меряет и llama-bench).
Медиана раундов; первый вызов каждого плеча выброшен.

Запуск: bench_scales_1024.py <файл.rwkvq> [раундов_префилла]
"""
import os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
T = int(os.environ.get("RWKVQ_T", "1024"))
NDEC = int(os.environ.get("RWKVQ_NDEC", "1024"))


def swap():
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()
    return float(o[5].rstrip("M"))


sw0 = swap()
m = qm.QuantRWKV7(load_raw(PATH))
AFF = os.environ.get("RWKVQ_AFFINE")   # "6,64": тяжёлые матрицы -> штатный MLX affine (23.09)
if AFF:
    # формат MLX-файлов MollySophia (int6, группа 64) В НАШЕМ движке: голова, r/k/v/o,
    # ffn -- mx.quantize + quantized_matmul; LoRA и эмбеддинг остаются нашей раскладкой
    sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")
    from eval_affine_inmem import to_affine
    _b, _g = (int(x) for x in AFF.split(","))
    to_affine(m, _b, _g); mx.clear_cache()
    PATH = PATH + " [affine %d/%d]" % (_b, _g)
IDX = mx.array(np.random.default_rng(0).integers(1, 60000, size=(1, T)).astype(np.int32))
step = mx.compile(m.forward_stateful)

pp = []
for r in range(ROUNDS + 1):
    st = m.init_state(1)
    mx.synchronize()
    t0 = time.perf_counter()
    lg, st2 = step(IDX, st, True)
    mx.eval(lg)
    mx.synchronize()
    dt = time.perf_counter() - t0
    if r:
        pp.append(dt)

dec = []
for r in range(2):
    st = m.init_state(1)
    lg, st = step(IDX, st, True)
    tok = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(tok, st)
    n = NDEC if r else 32
    mx.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        lg, st = step(tok[None], st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok, st)
    mx.synchronize()
    dt = time.perf_counter() - t0
    if r:
        dec.append((n, dt))

ppm = float(np.median(pp))
n, dt = dec[0]
print("%-34s pp%d %7.1f мс = %6.1f т/с (разброс %.1f%%) ; tg%d %8.3f мс/ток = %5.2f т/с ; своп %+.1f МБ ; активная %.0f МБ"
      % (os.path.basename(PATH), T, ppm * 1e3, T / ppm, 100 * (max(pp) - min(pp)) / ppm,
         n, dt / n * 1e3, n / dt, swap() - sw0, mx.get_active_memory() / 1e6), flush=True)
