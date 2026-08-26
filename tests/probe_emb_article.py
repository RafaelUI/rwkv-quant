"""СКОЛЬКО НА САМОМ ДЕЛЕ СТОИТ ПЛОТНЫЙ `emb` (приоритет 3, шаг 0).

Записано, что плотная таблица «снимает заодно половину остаточного пика
загрузки». Прежде чем писать под это Metal-кернель, статья меряется
аблацией -- ровно как `nodq` в rwkv-metal/tests/probe_train_mem.py: это
ВЕРХНЯЯ оценка приза, потому что убирает не только резидентность таблицы,
но и её транзиент при сборке.

  dense -- как сейчас: `_dense(emb.weight)` -> fp16 [V, D];
  gather -- боевой кандидат: строки деквантуются по требованию
           (quant_model.SymGatherEmb), плотной таблицы нет вовсе;
  stub  -- таблица не строится вовсе, а `emb_weight[idx]` отдаёт нули
           ТОЙ ЖЕ ФОРМЫ И ТОГО ЖЕ DTYPE (закон 34: заглушка, врущая
           типом, мерит ещё и тип -- на этом уже сгорела аблация WKV).

Пик снимается снаружи (`/usr/bin/time -l`, закон 22), аллокаторный --
внутри, отдельно после загрузки и после префилла. Своп печатается до и
после (закон 11).

    /usr/bin/time -l python tests/probe_emb_article.py <dense|stub|gather> <model.rwkvq> [T]
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.backends.metal import quant_model as qm  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

MODE = sys.argv[1]
PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/reduction_new.rwkvq"
T = int(sys.argv[3]) if len(sys.argv) > 3 else 512


def swap_mb():
    o = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                       capture_output=True, text=True,
                       env={**os.environ, "LC_ALL": "C"}).stdout
    p = o.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


class _Stub:
    """Вид таблицы: та же форма/dtype на выходе, без самой таблицы."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def __getitem__(self, idx):
        return mx.zeros(tuple(idx.shape) + (self.shape[1],), dtype=self.dtype)


def main():
    hit = {}
    if MODE == "gather":
        qm.EMB_GATHER = True
    if MODE == "stub":
        real = qm._dense

        def patched(qt):
            if getattr(qt, "key", "") == "emb.weight":
                hit["shape"] = tuple(qt.shape)
                # dtype ровно тот, что дал бы настоящий путь
                hit["dtype"] = (mx.float16 if len(qt.shape) == 2
                                and min(qt.shape) >= 32 else mx.float32)
                return _Stub(qt.shape, hit["dtype"])
            return real(qt)
        qm._dense = patched

    sw0 = swap_mb()
    t0 = time.time()
    model = QuantRWKV7(load_raw(PATH))
    load_s = time.time() - t0
    gc.collect(); mx.clear_cache()
    peak_load = mx.get_peak_memory() / 1e6

    # КОНТРОЛЬ ВКЛЮЧЕНИЯ: «заглушка не применилась» и «заглушка ничего не
    # дала» выглядят одинаково, поэтому подмена обязана быть видимой.
    kind = type(model.emb_weight).__name__
    want = {"stub": "_Stub", "gather": "SymGatherEmb", "dense": "array"}[MODE]
    assert kind == want, "режим %s, а emb_weight -- %s" % (MODE, kind)
    print("таблица: %s (%s %s)" % (kind, model.emb_weight.shape,
                                   model.emb_weight.dtype))

    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, T)).astype(np.int32))

    def prefill():
        st = model.init_state(1)
        logits, st = model.step(prompt, st, True)
        mx.eval(logits)
        return logits

    _warm = prefill(); del _warm      # прогрев компиляции
    mx.clear_cache()
    mx.reset_peak_memory()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        lg = prefill()
        ts.append(time.perf_counter() - t0)
        del lg
    dt = float(np.median(ts))
    sw1 = swap_mb()

    print("режим %s, %s, T=%d" % (MODE, os.path.basename(PATH), T))
    print("  загрузка %.1f с, аллокаторный пик после загрузки %.0f МБ"
          % (load_s, peak_load))
    print("  pp%d: %.1f ток/с (%.1f мс, разброс %.1f%%)"
          % (T, T / dt, dt * 1e3, 100 * (max(ts) - min(ts)) / dt))
    print("  аллокаторный пик за префилл %.0f МБ"
          % (mx.get_peak_memory() / 1e6))
    print("  своп %.0f -> %.0f МБ%s" % (sw0, sw1,
          "  *** ВЫРОС, замер недействителен ***" if sw1 > sw0 + 0.5 else ""))
    print("  footprint -- в /usr/bin/time -l снаружи")


if __name__ == "__main__":
    main()
