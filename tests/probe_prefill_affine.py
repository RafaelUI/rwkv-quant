"""ЧТО МОЖНО ВЫИГРАТЬ НА ПРЕФИЛЛЕ И ЧЕМ ЗА ЭТО ПЛАТИТЬ (разведка).

При N >= GEMM_MIN_BATCH_NB наш путь уходит с кернеля на `_dequant_w` +
плотный матмул: читает 6.5625 бит/вес, ПИШЕТ 16 бит/вес плотного fp16 и
читает их обратно -- впятеро больше трафика, чем нужно. Отсюда 1.82-2.38x
проигрыша на слое (замер 15.08) и 1.39x сквозным.

Вопрос этой разведки -- НЕ «быстрее ли нативный матмул» (это уже
измерено), а ЧЕМ ПЛАТИТЬ ЗА ВТОРУЮ РАСКЛАДКУ:

  1. какие group_size вообще поддержаны -- от этого зависит, возможен ли
     ТОЧНЫЙ репак sym (блок 16) или придётся квантовать заново;
  2. сколько стоит вторая раскладка в БАЙТАХ на разных gs;
  3. сколько теряется в ЗНАЧЕНИЯХ при переквантовании (двойное
     квантование: sym@6 -> деквант -> affine@k);
  4. и сколько из обещанного выигрыша остаётся на РЕАЛЬНЫХ формах при
     N=512.

Своп на границе; цепочки короткие -- на больших N неевалуированная
цепочка `_dequant_w` уже убивала машину (16 штук = 6.4 ГБ).

    python tests/probe_prefill_affine.py [model.rwkvq] [N] [раундов]
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
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def supported_group_sizes(bits):
    ok = []
    for gs in (16, 32, 64, 128):
        try:
            w = mx.random.normal((64, 512)).astype(mx.float16)
            q = mx.quantize(w, group_size=gs, bits=bits)
            x = mx.random.normal((2, 512)).astype(mx.float16)
            y = mx.quantized_matmul(x, q[0], scales=q[1], biases=q[2],
                                    transpose=True, group_size=gs, bits=bits)
            mx.eval(y)
            ok.append(gs)
        except Exception:
            pass
    return ok


def timeit(fn, reps=3):
    mx.eval(fn()); mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    sw0 = swap_mb()
    for bits in (6, 8):
        print(f"поддержанные group_size при {bits} битах: "
              f"{supported_group_sizes(bits)}")
    print()

    model = qm.QuantRWKV7(load_raw(PATH))
    tm, cm = model.blocks[1].tmix, model.blocks[1].cmix
    targets = [("cmix key [8192,2048]", cm.key),
               ("cmix value [2048,8192]", cm.value),
               ("proj r [2048,2048]", tm.r_proj),
               ("head [65536,2048]", model.head)]

    print(f"{'матрица':>24} | {'наш GEMM':>10} | вариант affine: время / "
          f"выигрыш / бит-вес / rel к точному декванту")
    for name, lin in targets:
        IN = lin.in_features
        x = mx.random.normal((N, IN)).astype(mx.float16)
        mx.eval(x)
        ours = timeit(lambda: lin(x))
        ref = lin._dequant_w()          # точный деквант нашей раскладки
        mx.eval(ref)
        y_ref = (x @ ref.T).astype(mx.float32)
        mx.eval(y_ref)
        print(f"{name:>24} | {ours*1e3:7.2f} мс |")
        for bits, gs in ((6, 32), (6, 64), (8, 32), (8, 64)):
            q = mx.quantize(ref, group_size=gs, bits=bits)
            mx.eval(q)
            t = timeit(lambda: mx.quantized_matmul(
                x, q[0], scales=q[1], biases=q[2], transpose=True,
                group_size=gs, bits=bits))
            y = mx.quantized_matmul(x, q[0], scales=q[1], biases=q[2],
                                    transpose=True, group_size=gs,
                                    bits=bits).astype(mx.float32)
            mx.eval(y)
            rel = float(mx.abs(y - y_ref).max() / mx.abs(y_ref).max())
            bpw = bits + 2 * 16 / gs
            print(f"{'':>24} |            | {bits} бит gs={gs:<3}: "
                  f"{t*1e3:6.2f} мс  {ours/t:5.2f}x  {bpw:5.3f} бит  "
                  f"rel {rel:.2e}")
            del q, y
        del ref, y_ref, x
    print(f"\nдля сравнения: наша раскладка sym@6 стоит 6.5625 бит/вес, "
          f"sym@8 -- 8.5625")
    print(f"своп: {sw0:.0f} -> {swap_mb():.0f} МБ")


if __name__ == "__main__":
    main()
