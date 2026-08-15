"""ЧЕСТНЫЙ МИКРОЗАМЕР ФОРМ LoRA: рабочий набор -- ВЕСЬ СТЕК, а не один слой.

ЗАЧЕМ ОТДЕЛЬНЫЙ ИНСТРУМЕНТ. `probe_lora_cost` мерил склейку на матрицах
ОДНОГО слоя (2.1 МБ) и получил 157.6 ГБ/с при измеренном потолке 104 --
то есть мерил системный кэш, а не DRAM. Здесь каждый вариант за один
проход обходит ВСЕ слои подряд, как это делает настоящий шаг: рабочий
набор 100 МБ, в кэш не влезает.

И ВТОРАЯ ПОПРАВКА, БЕЗ КОТОРОЙ СКЛЕЙКА -- ФАНТОМ. У четырёх down-проекций
РАЗНЫЕ входы: xw/xa/xv/xg -- это x, слитый с token-shift по СВОЕМУ
коэффициенту. Один матмул [512, D] с общим входом, который мерил прежний
probe, посчитал бы не ту величину. Точных способов склеить два:

  (з) z = [x, xx] и матрица [512, 2D] = [Acat | Acat*c]: поскольку
      x_i = x + xx*c_i, вклад лерпа складывается в матрицу. ОДИН запуск,
      но БАЙТ ВДВОЕ БОЛЬШЕ (у каждого ранга своя копия под xx).
  (б) батч по веткам с паддингом рангов до максимума (256): входы честные,
      но 4*256 = 1024 строки вместо 512 -- те же вдвое.

То есть склейка в fp16 стоит удвоения трафика, и вопрос ровно в том,
окупается ли оно укрупнением. Квантование меняет арифметику: при 6 битах
gs=64 удвоенная склейка (1.7 МБ/слой) ДЕШЕВЛЕ нынешнего fp16 порознь
(2.1 МБ/слой). Поэтому меряем оба рычага ПОРОЗНЬ и в комбинации.

Законы: 1 и 24 -- варианты чередуются ВНУТРИ раунда, печатается разброс
по раундам и отрыв победителя от второго; 11 -- своп на границе замера.

    python tests/probe_lora_shapes.py [model.rwkvq] [раундов]
"""
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
REPS = 3          # повторов замера внутри одного раунда на вариант
FP16 = mx.float16


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


class Variant:
    """Вариант раскладки: подготовленные буферы + функция одного прохода
    по ВСЕМ слоям. bytes_ -- сколько байт весов реально читается за проход."""

    def __init__(self, name, fn, nbytes, launches):
        self.name, self.fn, self.nbytes, self.launches = name, fn, nbytes, launches


def qbytes(w, bits, gs):
    """Байты affine-контейнера mx.quantize: коды bits/вес + scale и bias
    по fp16 на группу."""
    return w.size * (bits + 2 * 16 / gs) / 8


def build(model):
    """Собираем per-layer буферы всех вариантов. Слой 0 без v-ветки --
    берём слои 1.. (их 23 из 24), чтобы состав веток был одинаков."""
    layers = [b.tmix for b in model.blocks if b.tmix.v_lora_A is not None]
    D = model.n_embd
    x = mx.random.normal((1, D)).astype(mx.float32)
    xx = mx.random.normal((1, D)).astype(mx.float32)
    z = mx.concatenate([x, xx], axis=-1)
    xs = []          # честные лерпнутые входы для каждой ветки (w,a,v,g)
    for tm in layers:
        xs.append([(x + xx * c).astype(FP16)
                   for c in (tm.x_w.reshape(-1), tm.x_a.reshape(-1),
                             tm.x_v.reshape(-1), tm.x_g.reshape(-1))])
    x16, z16 = x.astype(FP16), z.astype(FP16)
    mx.eval(x, xx, z, x16, z16, [t for ts in xs for t in ts])

    A = [[tm.w_lora_A, tm.a_lora_A, tm.v_lora_A, tm.g_lora_A] for tm in layers]
    B = [[tm.w_lora_B_w, tm.a_lora_B_w, tm.v_lora_B_w, tm.g_lora_B_w]
         for tm in layers]
    coef = [[tm.x_w.reshape(1, -1), tm.x_a.reshape(1, -1),
             tm.x_v.reshape(1, -1), tm.x_g.reshape(1, -1)] for tm in layers]
    ranks = [a.shape[0] for a in A[0]]
    print(f"слоёв {len(layers)}, D={D}, ранги {ranks} (сумма {sum(ranks)})")

    out = []

    # ---------------- DOWN: как есть, четыре матмула, честные входы
    nb = sum(a.size for a in A[0]) * 2
    out.append(Variant(
        "down fp16 порознь (как сейчас)",
        lambda: [xs[i][j].astype(FP16) @ A[i][j].T
                 for i in range(len(layers)) for j in range(4)],
        nb, 4))

    # ---------------- DOWN: батч w/a/v с паддингом до 96 + g отдельно
    #                  (ровно то, что строит нынешний FUSE)
    wav = []
    for i in range(len(layers)):
        r = max(A[i][j].shape[0] for j in range(3))
        cols = []
        for j in range(3):
            a = A[i][j]
            if a.shape[0] < r:
                a = mx.concatenate(
                    [a, mx.zeros((r - a.shape[0], D), dtype=a.dtype)], axis=0)
            cols.append(a)
        wav.append(mx.contiguous(mx.stack(cols).transpose(0, 2, 1)))   # [3,D,r]
    mx.eval(wav)
    zs = [mx.stack([xs[i][0], xs[i][1], xs[i][2]]) for i in range(len(layers))]
    mx.eval(zs)
    nb = sum(w.size for w in wav[:1]) * 2 + A[0][3].size * 2
    out.append(Variant(
        "down fp16 батч wav+g (нынешний FUSE)",
        lambda: [y for i in range(len(layers))
                 for y in (zs[i] @ wav[i], xs[i][3].astype(FP16) @ A[i][3].T)],
        nb, 2))

    # ---------------- DOWN: склейка z=[x,xx], одна матрица [512, 2D]
    glue = []
    for i in range(len(layers)):
        parts = []
        for j in range(4):
            a = A[i][j]
            parts.append(mx.concatenate([a, (a.astype(mx.float32) *
                                             coef[i][j]).astype(a.dtype)],
                                        axis=1))                    # [r, 2D]
        glue.append(mx.contiguous(mx.concatenate(parts, axis=0)))    # [512, 2D]
    mx.eval(glue)
    nb = glue[0].size * 2
    out.append(Variant(
        "down fp16 склейка [512,2D]",
        lambda: [z16 @ glue[i].T for i in range(len(layers))],
        nb, 1))

    # ---------------- DOWN: та же склейка, но квантованная
    for bits in (8, 6):
        q = [mx.quantize(g, group_size=64, bits=bits) for g in glue]
        mx.eval([t for tr in q for t in tr])
        nb = qbytes(glue[0], bits, 64)
        out.append(Variant(
            f"down q{bits} склейка [512,2D] gs=64",
            (lambda q=q, bits=bits: lambda: [
                mx.quantized_matmul(z16, q[i][0], scales=q[i][1],
                                    biases=q[i][2], transpose=True,
                                    group_size=64, bits=bits)
                for i in range(len(layers))])(),
            nb, 1))

    # ---------------- DOWN: порознь, но квантованные (байты БЕЗ удвоения)
    for bits in (8, 6):
        q = [[mx.quantize(A[i][j], group_size=64, bits=bits) for j in range(4)]
             for i in range(len(layers))]
        mx.eval([t for row in q for tr in row for t in tr])
        nb = sum(qbytes(a, bits, 64) for a in A[0])
        out.append(Variant(
            f"down q{bits} порознь gs=64",
            (lambda q=q, bits=bits: lambda: [
                mx.quantized_matmul(xs[i][j], q[i][j][0], scales=q[i][j][1],
                                    biases=q[i][j][2], transpose=True,
                                    group_size=64, bits=bits)
                for i in range(len(layers)) for j in range(4)])(),
            nb, 4))

    # ---------------- UP: как есть, четыре матмула
    h = [[mx.random.normal((1, A[i][j].shape[0])).astype(FP16)
          for j in range(4)] for i in range(len(layers))]
    mx.eval([t for row in h for t in row])
    nb = sum(b.size for b in B[0]) * 2
    out.append(Variant(
        "up fp16 порознь (как сейчас)",
        lambda: [h[i][j] @ B[i][j].T
                 for i in range(len(layers)) for j in range(4)],
        nb, 4))

    # ---------------- UP: батч w/a/v (pad 96) + g -- нынешний FUSE
    bwav = []
    for i in range(len(layers)):
        r = max(B[i][j].shape[1] for j in range(3))
        rows = []
        for j in range(3):
            bt = B[i][j].T                                   # [r, D]
            if bt.shape[0] < r:
                bt = mx.concatenate(
                    [bt, mx.zeros((r - bt.shape[0], D), dtype=bt.dtype)], axis=0)
            rows.append(bt)
        bwav.append(mx.contiguous(mx.stack(rows)))           # [3,r,D]
    mx.eval(bwav)
    hwav = [mx.stack([h[i][0], h[i][1],
                      mx.concatenate([h[i][2], mx.zeros(
                          (1, bwav[i].shape[1] - h[i][2].shape[1]), dtype=FP16)],
                          axis=1)]) for i in range(len(layers))]
    mx.eval(hwav)
    nb = bwav[0].size * 2 + B[0][3].size * 2
    out.append(Variant(
        "up fp16 батч wav+g (нынешний FUSE)",
        lambda: [y for i in range(len(layers))
                 for y in (hwav[i] @ bwav[i], h[i][3] @ B[i][3].T)],
        nb, 2))

    # ---------------- UP: квантованные порознь, gs=32 (ранги 96/64/256)
    for bits in (8, 6):
        q = [[mx.quantize(B[i][j], group_size=32, bits=bits) for j in range(4)]
             for i in range(len(layers))]
        mx.eval([t for row in q for tr in row for t in tr])
        nb = sum(qbytes(b, bits, 32) for b in B[0])
        out.append(Variant(
            f"up q{bits} порознь gs=32",
            (lambda q=q, bits=bits: lambda: [
                mx.quantized_matmul(h[i][j], q[i][j][0], scales=q[i][j][1],
                                    biases=q[i][j][2], transpose=True,
                                    group_size=32, bits=bits)
                for i in range(len(layers)) for j in range(4)])(),
            nb, 4))

    return out, len(layers)


def time_pass(v, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(v.fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    sw0 = swap_mb()
    model = qm.QuantRWKV7(load_raw(PATH))
    variants, nl = build(model)
    print(f"своп после сборки: {sw0:.0f} -> {swap_mb():.0f} МБ\n")

    for v in variants:                      # прогрев всех веток
        mx.eval(v.fn())
    mx.synchronize()

    sw1 = swap_mb()
    res = {v.name: [] for v in variants}
    for r in range(ROUNDS):
        for v in variants:                  # ЧЕРЕДОВАНИЕ внутри раунда
            res[v.name].append(time_pass(v))
    sw2 = swap_mb()

    print(f"{'вариант':>38} | {'мс/слой':>9} {'разброс':>8} | "
          f"{'МБ/слой':>8} {'ГБ/с':>6} | зап.")
    print("-" * 92)
    for v in variants:
        ts = np.array(res[v.name]) / nl
        med = float(np.median(ts))
        spread = float((ts.max() - ts.min()) / med * 100)
        mb = v.nbytes / 1e6
        print(f"{v.name:>38} | {med*1e3:6.3f} мс {spread:6.1f}% | "
              f"{mb:8.3f} {mb/med/1e3:6.1f} | {v.launches}")

    print("\nсводка по стеку (мс на все слои, то есть вклад в шаг):")
    downs = [v for v in variants if v.name.startswith("down")]
    ups = [v for v in variants if v.name.startswith("up")]
    for group in (downs, ups):
        base = float(np.median(res[group[0].name]))
        order = sorted(group, key=lambda v: float(np.median(res[v.name])))
        for v in order:
            t = float(np.median(res[v.name]))
            print(f"  {v.name:>38} {t*1e3:7.3f} мс  "
                  f"{'база' if v is group[0] else f'{base/t:5.2f}x'}")
        if len(order) > 1:
            t1 = float(np.median(res[order[0].name]))
            t2 = float(np.median(res[order[1].name]))
            print(f"    отрыв победителя от второго: {(t2/t1-1)*100:.1f}%")

    print(f"\nсвоп: {sw1:.0f} -> {sw2:.0f} МБ "
          f"({'ОК' if sw2 - sw1 < 1 else 'ЗАМЕР НЕДЕЙСТВИТЕЛЕН'})")


if __name__ == "__main__":
    main()
