"""ЧТО ИМЕННО СТОИТ МИЛЛИСЕКУНД: покомпонентные аблации шага декода.

ЗАЧЕМ. Разложение вычитанием (`bench_step_decompose`) делит шаг надвое:
GEMV и «всё остальное». Внутри «остального» ~4 мс размазаны по десятку
разных вещей, и пока неизвестно, размазаны они РОВНО или там сидит один
дорогой компонент, который тормозит всё. Это разные диагнозы и разные
работы: в первом случае лечит только сокращение оп-каунта, во втором --
починка одного места.

МЕТОД. Компонент подменяется дешёвой заглушкой ДО `mx.compile` (иначе
компилятор оттрассирует старую ветку), полный и аблированный варианты
чередуются, берётся медиана разностей. Заглушка сохраняет ФОРМЫ, чтобы
граф оставался тем же по структуре.

ЧТО ЭТО НЕ МЕРИТ, И ЭТО ВАЖНО. Аблации НЕ АДДИТИВНЫ: убрав компонент, мы
меняем ещё и то, что с чем может перекрыться, поэтому сумма дельт не
обязана равняться шагу (ровно как вклады групп в KL -- см. раздел про
разложение по KL, там сумма изолированных дала 55-64% композита). Задача
здесь -- НАЙТИ ДОМИНАНТУ, а не построить точный бюджет.

Своп проверяется после каждой аблации: каждая делает свежий `mx.compile`,
и графы накапливаются.

    python tests/profile_components.py [model.rwkvq] [раундов]
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def main():
    # RWKVQ_FUSE=1 -- те же аблации на ФЬЮЗНУТОМ пути. Нужно, чтобы
    # отличить "компонент дорог сам по себе" от "компонент дорог потому,
    # что идёт россыпью мелких запусков": фьюз батчит LoRA (w,a,v) в два
    # матмула вместо шести и склеивает лерпы.
    qm.FUSE = os.environ.get("RWKVQ_FUSE") == "1"
    model = qm.QuantRWKV7(load_raw(PATH))
    D = model.n_embd
    sw0 = swap_mb()
    print(f"{os.path.basename(PATH)}, FUSE={qm.FUSE}, "
          f"своп на старте {sw0:.0f} МБ", flush=True)

    def bench(fn, n=25, warm=8):
        st = model.init_state(1)
        idx = mx.array(np.array([[123]], dtype=np.int32))
        logits, st = fn(idx, st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        for _ in range(warm):
            logits, st = fn(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(tok)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, st = fn(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(tok)
        mx.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    full = mx.compile(model.forward_stateful)
    base = bench(full)
    print(f"полный шаг: {base:.2f} мс/ток ({1000/base:.1f} ток/с)\n", flush=True)

    results = []

    def rebuild_fuse():
        """ФЬЮЗ КЕШИРУЕТ КОПИИ ВЕСОВ, и без сброса аблация его не задевает.

        `_build_fused` складывает конкатенацию r/k/v в `_rkv_fused` и стеки
        лор в `wav_At`/`wav_Bt` -- то есть подмена `tm.r_proj` или
        `tm.w_lora_A` меняет НЕФЬЮЗНУТУЮ ветку, а фьюзнутая продолжает
        считать по старым буферам. Первый прогон с FUSE=1 это и показал:
        дельта проекций упала с 3.89 до 1.26 мс, потому что реально
        аблировался один `o_proj`, а r/k/v считались по кешу. Числа были
        выброшены. Сброс флага заставляет пересобрать фьюз на аблированных
        весах."""
        if not qm.FUSE:
            return
        for blk in model.blocks:
            blk.tmix._fused_built = False

    def ab(name, apply, revert):
        deltas, fulls = [], []
        for _ in range(ROUNDS):
            rebuild_fuse()
            tA = bench(full)
            apply()
            rebuild_fuse()
            abl = mx.compile(model.forward_stateful)   # свежая трассировка
            tB = bench(abl)
            revert()
            rebuild_fuse()
            del abl
            fulls.append(tA)
            deltas.append(tA - tB)
        d = float(np.median(deltas))
        f = float(np.median(fulls))
        spread = max(deltas) - min(deltas)
        results.append((name, d, 100 * d / f, spread))
        print(f"{name:<28} {d:6.2f} мс  ({100*d/f:5.1f}% шага), "
              f"разброс дельты {spread:.2f} мс", flush=True)
        gc.collect()
        mx.clear_cache()
        sw = swap_mb()
        if sw > sw0 + 64:
            print(f"  СВОП ВЫРОС {sw0:.0f} -> {sw:.0f} МБ -- дальше замер "
                  f"недействителен, прерываю")
            sys.exit(1)

    # --- рекуррентность
    _wkv = qm._wkv_stateful
    ab("WKV-скан -> проброс",
       lambda: setattr(qm, "_wkv_stateful", lambda r, w, k, v, a, b, st: (v, st)),
       lambda: setattr(qm, "_wkv_stateful", _wkv))

    # --- нормировки
    _l2 = qm.l2_norm
    ab("l2_norm ключа -> ничего",
       lambda: setattr(qm, "l2_norm", lambda x: x),
       lambda: setattr(qm, "l2_norm", _l2))

    _gn = qm._group_norm
    ab("group_norm -> ничего",
       lambda: setattr(qm, "_group_norm", lambda x, H, w, b: x),
       lambda: setattr(qm, "_group_norm", _gn))

    _ln = qm._layer_norm
    ab("layer_norm (ln1/ln2/out) -> ничего",
       lambda: setattr(qm, "_layer_norm", lambda x, w, b, eps=1e-5: x),
       lambda: setattr(qm, "_layer_norm", _ln))

    # --- token shift
    _ts = qm._token_shift_stateful
    ab("token-shift -> ничего",
       lambda: setattr(qm, "_token_shift_stateful",
                       lambda x, prev: (x, x[:, -1:])),
       lambda: setattr(qm, "_token_shift_stateful", _ts))

    # --- LoRA-ветки: ранг 1 вместо 96/96/64/256
    saved = []

    def lora_off():
        for blk in model.blocks:
            tm = blk.tmix
            saved.append((tm.g_lora_A, tm.g_lora_B_w, tm.a_lora_A,
                          tm.a_lora_B_w, tm.w_lora_A, tm.w_lora_B_w,
                          getattr(tm, "v_lora_A", None),
                          getattr(tm, "v_lora_B_w", None)))
            for a, b in (("g_lora_A", "g_lora_B_w"), ("a_lora_A", "a_lora_B_w"),
                         ("w_lora_A", "w_lora_B_w")):
                setattr(tm, a, mx.zeros((1, D), dtype=mx.float16))
                setattr(tm, b, mx.zeros((D, 1), dtype=mx.float16))
            if getattr(tm, "v_lora_A", None) is not None:
                tm.v_lora_A = mx.zeros((1, D), dtype=mx.float16)
                tm.v_lora_B_w = mx.zeros((D, 1), dtype=mx.float16)

    def lora_on():
        for blk, s in zip(model.blocks, saved):
            tm = blk.tmix
            (tm.g_lora_A, tm.g_lora_B_w, tm.a_lora_A, tm.a_lora_B_w,
             tm.w_lora_A, tm.w_lora_B_w, vA, vB) = s
            if vA is not None:
                tm.v_lora_A, tm.v_lora_B_w = vA, vB
        saved.clear()

    ab("LoRA-ветки -> ранг 1", lora_off, lora_on)

    # --- проекции r/k/v/o: у них in == out, поэтому тождество сохраняет форму
    saved_p = []

    def proj_off():
        for blk in model.blocks:
            tm = blk.tmix
            saved_p.append((tm.r_proj, tm.k_proj, tm.v_proj, tm.o_proj))
            tm.r_proj = tm.k_proj = tm.v_proj = tm.o_proj = (lambda x: x)

    def proj_on():
        for blk, s in zip(model.blocks, saved_p):
            (blk.tmix.r_proj, blk.tmix.k_proj,
             blk.tmix.v_proj, blk.tmix.o_proj) = s
        saved_p.clear()

    ab("проекции r/k/v/o -> тождество", proj_off, proj_on)

    # --- cmix целиком
    saved_c = []

    class ZeroCM:
        def forward_stateful(self, x, ss):
            return x * 0.0, ss

    def cmix_off():
        for blk in model.blocks:
            saved_c.append(blk.cmix)
            blk.cmix = ZeroCM()

    def cmix_on():
        for blk, cm in zip(model.blocks, saved_c):
            blk.cmix = cm
        saved_c.clear()

    ab("cmix целиком -> ноль", cmix_off, cmix_on)

    # --- голова
    _head = model.head

    class TinyHead:
        def __call__(self, x):
            return x[..., :16]

    ab("голова -> обрезок",
       lambda: setattr(model, "head", TinyHead()),
       lambda: setattr(model, "head", _head))

    results.sort(key=lambda r: -r[1])
    print(f"\n{'компонент':<28} {'мс':>7} {'% шага':>8}")
    for name, d, pct, _ in results:
        print(f"{name:<28} {d:7.2f} {pct:7.1f}%")
    print(f"\nсумма дельт {sum(r[1] for r in results):.2f} мс при шаге "
          f"{base:.2f} -- аддитивности НЕТ и быть не должно, см. докстринг")
    print(f"своп: {sw0:.0f} -> {swap_mb():.0f} МБ")


if __name__ == "__main__":
    main()
