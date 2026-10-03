"""04.10: кривая скорость--качество (для статьи и Bo Peng): несколько файлов ОДНОЙ модели в одном процессе, плечи чередуются
по раундам с циклическим сдвигом и разворотом порядка (закон 1 / 41), контроль A/A -- то же плечо под вторым именем.
Декод -- рабочий путь generate_step(pipeline=True) (конвейер async_eval), бёрст N токенов с живым state;
префилл -- forward_stateful(last_only=True) под mx.compile на T токенах (как bench_scales_1024).
Итог: медиана по раундам и ПАРНОЕ отношение к опорному плечу (первое в списке) с бутстрэпом по раундам; своп до/после.
    python speed_curve_0410.py <out.json> <имя=файл.rwkvq[@affine6]> ...     (affine6: тяжёлые матрицы -> mx affine 6/64 в памяти)
    окружение: SC_R (раундов декода, 12), SC_N (токенов в бёрсте, 32), SC_RP (раундов префилла, 6), SC_T (1024)"""
import json, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant"); sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")
import numpy as np, torch
import mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal import generate as G
OUT = sys.argv[1]; SPEC = [a.split("=", 1) for a in sys.argv[2:]]
R, N, RP, T = (int(os.environ.get(k, d)) for k, d in (("SC_R", 12), ("SC_N", 32), ("SC_RP", 6), ("SC_T", 1024)))
sw = lambda: float(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()[5].rstrip("M"))
sw0 = sw(); t00 = time.time()
M, info = {}, {}
for name, path in SPEC:
    aff = path.endswith("@affine6"); path = path[:-8] if aff else path
    m = QuantRWKV7(load_raw(path))
    if aff:
        from eval_affine_inmem import to_affine
        to_affine(m, 6, 64); mx.clear_cache()
    M[name] = m; info[name] = dict(file=os.path.basename(path), bytes=os.path.getsize(path), affine=aff, fast_ln=bool(m.fast_ln))
names = list(M); ref = names[0]
arms = names + [ref + "#AA"]; M[ref + "#AA"] = M[ref]
P = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt"), weights_only=False)["tokens"][0, :64].tolist()
IDX = mx.array(np.random.default_rng(0).integers(1, 60000, size=(1, T)).astype(np.int32))
ST, STEP = {}, {}
def dec(name):
    m = M[name]; st, last = ST.get(name, (None, P)); out = []
    t0 = time.perf_counter()
    for y, _ in G.generate_step(m, last, N, pipeline=True, state=st, out=out):
        pass
    dt = (time.perf_counter() - t0) / N
    ST[name] = (out[0], [int(y.item())])
    return dt * 1e3
def pre(name):
    m = M[name]
    if name.split("#")[0] not in STEP: STEP[name.split("#")[0]] = mx.compile(m.forward_stateful)
    step = STEP[name.split("#")[0]]; st = m.init_state(1); mx.synchronize()
    t0 = time.perf_counter(); lg, _ = step(IDX, st, True); mx.eval(lg); mx.synchronize()
    return T / (time.perf_counter() - t0)
def order(r):
    k = r % len(arms); o = arms[k:] + arms[:k]
    return o[::-1] if (r // len(arms)) % 2 else o
for a in arms: dec(a); dec(a)                      # прогрев (первый бёрст включает префилл промпта)
sw1 = sw(); t0 = time.time()
D = {a: [] for a in arms}
for r in range(R):
    for a in order(r): D[a].append(dec(a))
t_dec = time.time() - t0
for a in arms: pre(a)                               # компиляция и первый вызов выброшены
t0 = time.time(); Pp = {a: [] for a in arms}
for r in range(RP):
    for a in order(r): Pp[a].append(pre(a))
t_pre = time.time() - t0; sw2 = sw()
rng = np.random.default_rng(0)
def rel(x, y, inv=False):
    x, y = np.array(x), np.array(y); d = (y / x) if not inv else (x / y)   # > 1 -- быстрее опорного
    b = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
res = dict(info=info, R=R, N=N, RP=RP, T=T, swap_before=sw0, swap_loaded=sw1, swap_after=sw2, active_mb=mx.get_active_memory() / 1e6,
           dec_ms=D, pre_tps=Pp, sec_dec=t_dec, sec_pre=t_pre)
json.dump(res, open(OUT, "w"))
print("опорное плечо %s; декод %d x %d ток (%.0f с), префилл %d x %d ток (%.0f с); своп: загрузка %+.1f МБ, замер %+.1f МБ; активная %.0f МБ" % (
    ref, R, N, t_dec, RP, T, t_pre, sw1 - sw0, sw2 - sw1, res["active_mb"]), flush=True)
for a in arms:
    d = np.array(D[a]); p = np.array(Pp[a]); h = len(d) // 2
    rd = rel(D[ref], D[a], inv=True); rp = rel(Pp[ref], Pp[a])
    print("%-10s %7.1f МБ | декод %6.2f т/с (мс/ток медиана %.3f; 1-я половина %.3f, 2-я %.3f) x%.3f [%.3f; %.3f] | префилл %7.1f т/с (разброс %.1f%%) x%.3f [%.3f; %.3f]" % (
        a, info[a.split("#")[0]]["bytes"] / 1e6, 1e3 / np.median(d), np.median(d), np.median(d[:h]), np.median(d[h:]), *rd,
        np.median(p), 100 * (p.max() - p.min()) / np.median(p), *rp), flush=True)
print("всего %.0f с" % (time.time() - t00), flush=True)
