"""A/B цены пред-WKV: настоящий шаг с ядром и без, ЧЕРЕДОВАНИЕМ.

ПОЧЕМУ ЧЕРЕДОВАНИЕМ И В ОДНОМ ПРОЦЕССЕ. 10.09 доля WKV, снятая отдельным
процессом, разошлась вдвое из-за троттлинга безвентиляторного корпуса
(закон 25). Два плеча здесь идут вперемешку по раундам, поэтому тепловая
история у них общая, а не последовательная.

ЛОВУШКА COMPILE. step трассирует ветку на момент mx.compile, и смена
FUSE_PREWKV после компиляции НИЧЕГО НЕ МЕНЯЕТ -- померилось бы одно и то
же плечо дважды. Поэтому перед каждым замером кеш компиляции сбрасывается
явно, а WARM прогоняется уже после сброса.
"""
import gc, os, subprocess, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.backends.metal import fused_prewkv as fp
from rwkv_quant.formats.reader import load_raw

PATH = sys.argv[1]
ROUNDS = int(os.environ.get("RWKVQ_AB_ROUNDS", "9"))
REPS = int(os.environ.get("RWKVQ_AB_REPS", "7"))
WARM = 3


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]
    unit, num = u[-1], u[:-1]
    return float(num) * (1024 if unit == "G" else 1)


# ТРЕТЬЕ ПЛЕЧО -- ЗАГЛУШКА. Она НЕВЕРНА численно и нужна только чтобы
# узнать ПОТОЛОК выигрыша: сколько стоит арифметика пред-WKV в настоящем
# компилированном шаге, если убрать её целиком. Если заглушка не
# отличается от плеча без ядра, статья 0.387 мс в шаге не живёт, и
# фьюзить там нечего -- сколько ни пиши кернелей.
# Reshape вместо математики, чтобы y_w и y_a остались живыми и compile
# не выкинул ветки LoRA как мёртвые (иначе выигрыш будет завышен чужой
# статьёй).
_REF = qm.QuantTMix._prewkv


def _stub(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype):
    shp = (B, T, self.H, self.S)
    w = y_w.reshape(shp)
    a = y_a.reshape(shp)
    return w, k, v, k, a, (v if self.layer_id == 0 else v_first)


def _outside(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype):
    # РАЗДЕЛЯЮЩЕЕ ПЛЕЧО: делает ровно ту обвязку, которую ядро оставляет
    # снаружи (три сложения и два sigmoid), а дальше возвращает мусор.
    # Разница с заглушкой -- ЦЕНА ОДНОЙ ОБВЯЗКИ, без арифметики ядра.
    shp = (B, T, self.H, self.S)
    zw = y_w + self.w_lora_B_b
    sa = mx.sigmoid(y_a + self.a_lora_B_b)
    if self.layer_id == 0:
        acc = zw + sa
    else:
        acc = zw + sa + mx.sigmoid(y_v + self.v_lora_B_b)
    return acc.reshape(shp), k, v, k, acc.reshape(shp), (
        v if self.layer_id == 0 else v_first)


def _inside(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype):
    # ЯДРО С sigmoid ВНУТРИ: численно не проходит бит-в-бит на k и v,
    # существует только чтобы узнать потолок скорости этой конструкции.
    return fp.prewkv_kernel_inside(
        y_w, y_a, y_v, k, v, v_first, self.w_lora_B_b, self.a_lora_B_b,
        self.v_lora_B_b, self.k_k, self.k_a, self.layer_id, B, T,
        self.H, self.S, dtype)


def _empty(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype):
    # Голый запуск той же геометрии, без математики.
    return fp.prewkv_kernel_empty(
        y_w, y_a, y_v, k, v, v_first, self.w_lora_B_b, self.a_lora_B_b,
        self.v_lora_B_b, self.k_k, self.k_a, self.layer_id, B, T,
        self.H, self.S, dtype)


def _simd(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype):
    return fp.prewkv_kernel_simd(
        y_w, y_a, y_v, k, v, v_first, self.w_lora_B_b, self.a_lora_B_b,
        self.v_lora_B_b, self.k_k, self.k_a, self.layer_id, B, T,
        self.H, self.S, dtype)


_MODES = {"stub": _stub, "outside": _outside, "inside": _inside,
          "simd": _simd,

          "empty": _empty}



def measure(m, state, mode):
    qm.QuantTMix._prewkv = _MODES.get(mode, _REF)
    qm.FUSE_PREWKV = (mode == "kernel")
    if hasattr(m, "_step_compiled"):
        del m._step_compiled
    step = m.step

    def full():
        lg, state["st"] = step(state["tok"][None], state["st"])
        state["tok"] = mx.argmax(lg[:, -1], axis=-1)
        return state["tok"]

    for _ in range(WARM):
        mx.eval(full())
    mx.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(full())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    sw0 = swap_mb()
    m = qm.QuantRWKV7(load_raw(PATH))
    print("файл: %s" % os.path.basename(PATH), flush=True)
    tok = mx.array(np.array([1], dtype=np.int32))
    st = m.init_state(1)
    qm.FUSE_PREWKV = False
    lg, st = m.forward_stateful(mx.array(np.arange(1, 65, dtype=np.int32))[None],
                                st, last_only=True)
    mx.eval(lg)
    t0 = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(t0)
    gc.collect()
    mx.clear_cache()

    A = {"st": st, "tok": t0}
    B = {"st": st, "tok": t0}
    C = {"st": st, "tok": t0}
    E = {"st": st, "tok": t0}
    F = {"st": st, "tok": t0}
    G = {"st": st, "tok": t0}
    Hh = {"st": st, "tok": t0}
    ra, rb, rc, rd, re_, rg, rh = [], [], [], [], [], [], []
    for i in range(ROUNDS):
        a = measure(m, A, "off")
        b = measure(m, B, "kernel")
        c = measure(m, C, "stub")
        d = measure(m, E, "outside")
        e = measure(m, F, "inside")
        g = measure(m, G, "empty")
        hh = measure(m, Hh, "simd")
        ra.append(a)
        rb.append(b)
        rc.append(c)
        rd.append(d)
        re_.append(e)
        rg.append(g)
        rh.append(hh)
        print("раунд %d: без %.3f, ядро %.3f, заглушка %.3f, "
              "обвязка %.3f, ядро+sigmoid %.3f, пустое %.3f, simd %.3f" % (
                  i, a, b, c, d, e, g, hh), flush=True)
    qm.QuantTMix._prewkv = _REF
    ma, mb = float(np.median(ra)), float(np.median(rb))
    mc = float(np.median(rc))
    print("-" * 60, flush=True)
    print("без ядра  %.3f мс = %.1f т/с" % (ma, 1000.0 / ma), flush=True)
    print("с ядром   %.3f мс = %.1f т/с" % (mb, 1000.0 / mb), flush=True)
    print("снято     %+.3f мс, %+.1f т/с, %+.2f%%" % (
        ma - mb, 1000.0 / mb - 1000.0 / ma, 100.0 * (ma - mb) / ma), flush=True)
    print("заглушка %.3f мс = %.1f т/с -- ПОТОЛОК, численно неверна" % (
        mc, 1000.0 / mc), flush=True)
    print("потолок статьи: %+.3f мс (без ядра минус заглушка)" % (
        ma - mc), flush=True)
    md, me = float(np.median(rd)), float(np.median(re_))
    print("обвязка без ядра   %.3f мс -- цена трёх сложений и двух sigmoid"
          % md, flush=True)
    print("ядро с sigmoid     %.3f мс = %.1f т/с, снято %+.3f мс" % (
        me, 1000.0 / me, ma - me), flush=True)
    mg = float(np.median(rg))
    print("пустое ядро        %.3f мс -- цена запуска без арифметики,"
          " сверх заглушки %+.3f мс" % (mg, mg - mc), flush=True)
    mh = float(np.median(rh))
    print("ядро с simd-суммой  %.3f мс = %.1f т/с, снято %+.3f мс" % (
        mh, 1000.0 / mh, ma - mh), flush=True)
    print("разброс раундов: без ядра %.3f..%.3f, с ядром %.3f..%.3f" % (
        min(ra), max(ra), min(rb), max(rb)), flush=True)
    print("своп дельта %+.1f МБ" % (swap_mb() - sw0), flush=True)


main()
