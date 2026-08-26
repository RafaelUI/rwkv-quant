"""A/B: emb-gather против плотной таблицы -- декод и префилл, ОДИН процесс.

Статья чисто резидентная, поэтому решает не она, а цена в шаге: декод
платит за КОЛИЧЕСТВО запусков (очередь CPU->GPU 2.8 мс), а gather
добавляет на токен цепочку MLX-операций. Если она заметна -- вариант (а)
остаётся запасным путём, а не боевым.

Чередование в одном процессе с рандомизацией порядка (законы 1, 24), своп
по раундам (закон 11). Обе таблицы живут ОДНОВРЕМЕННО, переключается
только ссылка `model.emb_weight`; скомпилированный шаг сбрасывается при
каждом переключении, иначе второе плечо считалось бы графом первого.

КОНТРОЛЬ ВКЛЮЧЕНИЯ ОБЯЗАТЕЛЕН И НЕ МОЖЕТ БЫТЬ ПО ВЫХОДУ: пути равны
БИТ-В-БИТ (tests/test_emb_gather_parity.py), поэтому "плечо не
переключилось" и "переключилось, но цены нет" по логитам неразличимы.
Контроль структурный -- тип объекта таблицы на каждом плече.

    python tests/bench_emb_gather_ab.py [model.rwkvq] [раундов] [токенов]
"""
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

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_new.rwkvq"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 32
TPRE = 512


def swap_mb():
    o = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                       capture_output=True, text=True,
                       env={**os.environ, "LC_ALL": "C"}).stdout
    p = o.replace("=", " ").split()
    return float(p[p.index("used") + 1].rstrip("M"))


def main():
    ck = load_raw(PATH)
    qt = ck.tensors["emb.weight"]
    model = QuantRWKV7(ck)
    dense_tbl = model.emb_weight
    qm.EMB_GATHER = True
    gather_tbl = qm._emb_table(qt)
    assert isinstance(gather_tbl, qm.SymGatherEmb)
    assert not isinstance(dense_tbl, qm.SymGatherEmb)

    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, TPRE)).astype(np.int32))
    toks = [mx.array(rng.integers(1, 60000, size=(1, 1)).astype(np.int32))
            for _ in range(NTOK)]

    def use(tbl):
        model.emb_weight = tbl
        if hasattr(model, "_step_compiled"):
            del model._step_compiled     # иначе второе плечо -- граф первого

    def decode():
        st = model.init_state(1)
        lg, st = model.step(prompt[:, :1], st, True)
        mx.eval(lg, *[x for x in st if x is not None])
        for t in toks:
            lg, st = model.step(t, st, True)
        mx.eval(lg)

    def prefill():
        st = model.init_state(1)
        lg, st = model.step(prompt, st, True)
        mx.eval(lg)

    VAR = [("плотная", dense_tbl), ("gather", gather_tbl)]
    dec = {n: [] for n, _ in VAR}
    pre = {n: [] for n, _ in VAR}
    pk = {n: 0.0 for n, _ in VAR}
    seen = {n: "" for n, _ in VAR}

    for name, tbl in VAR:            # прогрев обоих плеч до замеров
        use(tbl); decode(); prefill()

    sw0 = swap_mb()
    rs = np.random.RandomState(23)
    for rnd in range(ROUNDS):
        order = VAR if rnd % 2 == 0 else VAR[::-1]
        if rs.rand() < 0.5:
            order = order[::-1]
        for name, tbl in order:
            use(tbl)
            seen[name] = type(model.emb_weight).__name__
            mx.clear_cache(); mx.reset_peak_memory()
            t0 = time.perf_counter(); decode()
            dec[name].append((time.perf_counter() - t0) / NTOK)
            t0 = time.perf_counter(); prefill()
            pre[name].append(time.perf_counter() - t0)
            pk[name] = max(pk[name], mx.get_peak_memory() / 1e6)
        print("  раунд %d: своп %.0f МБ" % (rnd, swap_mb()), flush=True)
    sw1 = swap_mb()
    model.emb_weight = dense_tbl

    assert seen["плотная"] != seen["gather"], (
        "оба плеча видели одну таблицу: %s" % seen)
    print("\nконтроль включения: плотная -> %s, gather -> %s"
          % (seen["плотная"], seen["gather"]))
    print("%s, раундов %d, токенов на замер %d\n"
          % (os.path.basename(PATH), ROUNDS, NTOK))
    print("%-10s %11s %9s %11s %9s %9s"
          % ("плечо", "декод мс/ток", "ток/с", "pp512 мс", "ток/с", "пик МБ"))
    med = {}
    for name, _ in VAR:
        d = float(np.median(dec[name])) * 1e3
        p = float(np.median(pre[name])) * 1e3
        med[name] = (d, p)
        sp = (max(dec[name]) - min(dec[name])) / float(np.median(dec[name])) * 100
        print("%-10s %11.2f %9.1f %11.1f %9.1f %9.0f   (разброс декода %.1f%%)"
              % (name, d, 1e3 / d, p, TPRE / p * 1e3, pk[name], sp))
    dd = med["gather"][0] - med["плотная"][0]
    dp = med["gather"][1] - med["плотная"][1]
    print("\nцена gather: декод %+.2f мс/ток (%+.1f%%), префилл %+.1f мс (%+.1f%%)"
          % (dd, 100 * dd / med["плотная"][0], dp, 100 * dp / med["плотная"][1]))
    print("своп %.0f -> %.0f МБ%s" % (sw0, sw1,
          "  *** ВЫРОС, замер недействителен ***" if sw1 > sw0 + 0.5 else ""))


if __name__ == "__main__":
    main()
