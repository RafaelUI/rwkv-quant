"""СКВОЗНОЙ A/B СТЕЙДЖИНГА В WKV: префилл и декод, чередование в одном процессе.

Арбитр для микрозамеров `bench_wkv_acc`: те греют кэш повторами и запрещают
перекрытие, здесь -- настоящий стек.

КАК СДЕЛАН A/B. Прежнее ядро восстановлено ЗДЕСЬ ЖЕ (замороженная копия
тела до правки), обе ветки живут в одном процессе, одном венве и одном
состоянии кеша (закон 27). Тонкость: `mx.compile` трассирует ветку НА
МОМЕНТ КОМПИЛЯЦИИ, поэтому подмена `qm._wkv_stateful` после трассировки не
видна -- сначала патчим и компилируем, ПОТОМ чередуем уже готовые
скомпилированные объекты, и подмена больше ни на что не влияет.

ФАКТ ВКЛЮЧЕНИЯ. Правка бит-в-бит, поэтому по выходу две ветки НЕ различить,
и обычный «выход изменился» тут не сработал бы. Вместо него печатается
сверка исходников ядер (обязаны отличаться) и равенство логитов (обязано
быть точным) -- второе и есть сквозная проверка правки.

    python bench_wkv_e2e_ab.py [model.rwkvq] [T] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402
import importlib  # noqa: E402
# rwkv_metal.kernel.__init__ переопределяет имя wkv7 ФУНКЦИЕЙ, поэтому
# обычный import отдаёт её, а не модуль.
wk = importlib.import_module("rwkv_metal.kernel.wkv7")  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# --- ЗАМОРОЖЕННАЯ КОПИЯ ЯДРА ДО ПРАВКИ ------------------------------------
OLD_BODY = r"""
    uint dv   = thread_position_in_grid.y;
    uint bhi  = thread_position_in_grid.x;
    uint bi   = bhi / H_C; uint hi = bhi % H_C;

    float h_row[HEAD_SIZE_C];
    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];

    for (uint t=0; t<CHUNK_C; t++) {
        uint base = ((bi*CHUNK_C+t)*H_C+hi)*HEAD_SIZE_C;
        float sa = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];
        float v_dv = v[base+dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            h_row[dk] = w[base+dk]*h_row[dk] + v_dv*k[base+dk] + sa*b[base+dk];
        float y = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
        out[((bi*CHUNK_C+t)*H_C+hi)*HEAD_SIZE_C+dv] = y;
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
"""
_old_cache = {}


def _get_old(H, Tk):
    key = (H, Tk)
    if key in _old_cache:
        return _old_cache[key]
    hdr = (f"\nconstant uint HEAD_SIZE_C = {wk.HEAD_SIZE};"
           f"\nconstant uint CHUNK_C     = {Tk};"
           f"\nconstant uint H_C         = {H};\n")
    _old_cache[key] = mx.fast.metal_kernel(
        name=f"wkv7_infer_OLD_{H}_{Tk}",
        input_names=["r", "w", "k", "v", "a", "b", "h_in"],
        output_names=["out", "h_out"],
        header=hdr, source=OLD_BODY,
    )
    return _old_cache[key]


def wkv_old(r, w, k, v, a, b, h):
    B, Tk, H, D = r.shape
    res = _get_old(H, Tk)(
        inputs=[x.astype(mx.float32) for x in [r, w, k, v, a, b, h]],
        grid=(B * H, D, 1), threadgroup=(1, 1, 1),
        output_shapes=[(B, Tk, H, D), (B, H, D, D)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return res[0], res[1]


def swap_mb():
    env = dict(os.environ, LC_ALL="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                         text=True, env=env).stdout.replace("=", " ").split()
    return float(out[out.index("used") + 1].rstrip("M").replace(",", "."))


def main():
    import inspect
    cur = inspect.getsource(wk._get_infer_kernel)
    print(f"ядро в дереве содержит стейджинг: "
          f"{'ДА' if 'threadgroup float a_sh' in cur else 'НЕТ -- мерить нечего'}; "
          f"замороженная копия -- нет: "
          f"{'ДА' if 'threadgroup float' not in OLD_BODY else 'НЕТ'}")
    model = qm.QuantRWKV7(load_raw(PATH))
    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))
    tok1 = mx.array(np.array([[1]], dtype=np.int32))

    new_fn = qm._wkv_stateful

    def build(fn):
        """Патчим -> компилируем -> прогреваем ОБЕ формы под этим патчем."""
        qm._wkv_stateful = fn
        comp = mx.compile(model.forward_stateful)
        mx.eval(comp(prompt, model.init_state(1), True)[0])
        mx.eval(comp(tok1, model.init_state(1))[0])
        mx.synchronize()
        return comp

    comp_new = build(new_fn)
    comp_old = build(wkv_old)
    qm._wkv_stateful = new_fn          # дальше подмена уже ни на что не влияет

    # --- контроль: логиты обязаны совпасть ТОЧНО ---------------------------
    lo_new = np.asarray(comp_new(prompt, model.init_state(1), True)[0])
    lo_old = np.asarray(comp_old(prompt, model.init_state(1), True)[0])
    d = float(np.abs(lo_new - lo_old).max())
    print(f"логиты префилла: max|Δ| = {d:.3e} "
          f"({'РАВЕНСТВО' if d == 0 else 'РАСХОЖДЕНИЕ -- разбираться'}), "
          f"амплитуда {np.abs(lo_old).max():.1f}")

    def prefill(fn):
        mx.synchronize()
        t0 = time.perf_counter()
        lo, _ = fn(prompt, model.init_state(1), True)
        mx.eval(lo)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    def decode(fn, n=24):
        st = model.init_state(1)
        lo, st = fn(prompt, st, True)
        tok = mx.argmax(lo[:, -1], axis=-1)
        mx.eval(tok)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            lo, st = fn(tok[None], st)
            tok = mx.argmax(lo[:, -1], axis=-1)
            mx.eval(tok)
        mx.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    for _ in range(2):
        prefill(comp_old); prefill(comp_new)

    sw0 = swap_mb()
    pre = {"было": [], "стало": []}
    dec = {"было": [], "стало": []}
    for _ in range(ROUNDS):
        pre["было"].append(prefill(comp_old))
        pre["стало"].append(prefill(comp_new))
    for _ in range(ROUNDS):
        dec["было"].append(decode(comp_old))
        dec["стало"].append(decode(comp_new))
    sw1 = swap_mb()

    print(f"\nсвоп {sw0:.0f} -> {sw1:.0f} МБ "
          f"({'валиден' if abs(sw1 - sw0) < 1 else 'НЕДЕЙСТВИТЕЛЕН, закон 11'})")

    def rep(name, d, unit):
        a = float(np.median(d["было"])); b = float(np.median(d["стало"]))
        sa = 100 * (max(d["было"]) - min(d["было"])) / a
        sb = 100 * (max(d["стало"]) - min(d["стало"])) / b
        wins = sum(x < y for x, y in zip(d["стало"], d["было"]))
        print(f"{name}: было {a:8.2f} {unit} (разброс {sa:4.1f}%) | "
              f"стало {b:8.2f} {unit} (разброс {sb:4.1f}%) | "
              f"{a/b:.3f}x, {a-b:+.2f} {unit}, пар выиграно {wins}/{len(d['было'])}")
        return a, b

    a, b = rep(f"префилл pp{T}", pre, "мс")
    print(f"              {T/a*1e3:.1f} -> {T/b*1e3:.1f} ток/с")
    a, b = rep("декод       ", dec, "мс/ток")
    print(f"              {1000/a:.2f} -> {1000/b:.2f} ток/с")


if __name__ == "__main__":
    main()
