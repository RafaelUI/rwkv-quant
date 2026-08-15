"""СКВОЗНОЙ A/B КВАНТОВАНИЯ LoRA-ВЕТОК: декод и префилл, чередованием.

ЗАЧЕМ СКВОЗНОЙ, ЕСЛИ ЕСТЬ МИКРОЗАМЕР. Микрозамеры на этих формах врут в
ОБЕ стороны и по разным причинам. `probe_lora_cost` мерил склейку на
матрицах одного слоя -- 157.6 ГБ/с при потолке 104, то есть кэш.
`probe_lora_shapes` чинит рабочий набор (все слои), но даёт 23 слоя
НЕЗАВИСИМОЙ работы, которую GPU перекрывает, тогда как настоящий шаг --
цепочка зависимостей: там цена запуска прячется хуже. Первый завышает
выигрыш склейки, второй занижает. Арбитр -- только этот замер.

ОДИН ПРОЦЕСС, ОДНА МОДЕЛЬ, ОДНА БИТНОСТЬ. Буферы sep и glue строятся
вместе и живут одновременно, поэтому между бёрстами ничего не
перестраивается и состояние памяти не меняется (иначе сравнивались бы
разные состояния машины, а не раскладки). Битность меняет буферы -- она
задаётся аргументом и на процесс одна.

    python tests/bench_lora_quant_ab.py [model.rwkvq] [бит] [раундов]
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
BITS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NSTEP = 25
PREFILL_T = 512
FUSE = os.environ.get("RWKVQ_FUSE", "0") == "1"


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
    qm.FUSE = FUSE
    model = qm.QuantRWKV7(load_raw(PATH))
    qm.LORA_QBITS = BITS
    qm.LORA_Q = "sep"
    qm.reset_lora_q(model)
    for b in model.blocks:                      # буферы обеих раскладок
        b.tmix._build_lora_q()
    built = sum(b.tmix._lq_A is not None for b in model.blocks)
    assert built == len(model.blocks), f"буферы не построены: {built}"

    # RWKVQ_MODES=None,sep -- сузить набор. На 2.9B это не косметика:
    # буферы склейки живут одновременно с поветочными, а склейка уже
    # отвергнута замером, и платить за неё памятью на большом чекпоинте
    # незачем.
    _m = os.environ.get("RWKVQ_MODES")
    modes = ([None if m == "None" else m for m in _m.split(",")] if _m
             else [None, "sep", "glue"])
    fns = {}
    for m in modes:                             # своя трассировка на вариант
        qm.LORA_Q = m
        fns[m] = mx.compile(model.forward_stateful)

    idx1 = mx.array(np.array([[187]], dtype=np.int32))
    idxP = mx.array(np.random.randint(0, 65500, size=(1, PREFILL_T),
                                      dtype=np.int32))

    def decode_burst(fn, mode, n=NSTEP):
        qm.LORA_Q = mode
        st = model.init_state(1)
        logits, st = fn(idx1, st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        for _ in range(6):                      # прогрев/трассировка
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

    def prefill_once(fn, mode):
        qm.LORA_Q = mode
        st = model.init_state(1)
        logits, st = fn(idxP, st)
        mx.eval(logits)
        mx.synchronize()
        st = model.init_state(1)
        t0 = time.perf_counter()
        logits, st = fn(idxP, st)
        mx.eval(logits)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for m in modes:                             # прогрев всех трассировок
        decode_burst(fns[m], m, n=3)
        prefill_once(fns[m], m)

    sw0 = swap_mb()
    dec = {m: [] for m in modes}
    pre = {m: [] for m in modes}
    for r in range(ROUNDS):
        for m in modes:                         # ЧЕРЕДОВАНИЕ внутри раунда
            dec[m].append(decode_burst(fns[m], m))
        for m in modes:
            pre[m].append(prefill_once(fns[m], m))
        print(f"  раунд {r}: " + "  ".join(
            f"{str(m):>4} {dec[m][-1]:.2f} мс/ток / {pre[m][-1]:.0f} мс pp"
            for m in modes) + f" | своп {swap_mb():.0f}", flush=True)
    sw1 = swap_mb()

    print(f"\nмодель {os.path.basename(PATH)}, LoRA {BITS} бит, "
          f"FUSE={FUSE}, {ROUNDS} раундов по {NSTEP} шагов")
    print(f"{'вариант':>8} | {'мс/ток':>8} {'ток/с':>7} {'разброс':>8} "
          f"{'против fp16':>12} | {'pp512 мс':>9} {'ток/с':>7} {'против':>8}")
    b_d = float(np.median(dec[None]))
    b_p = float(np.median(pre[None]))
    for m in modes:
        d = np.array(dec[m]); p = np.array(pre[m])
        md, mp = float(np.median(d)), float(np.median(p))
        print(f"{str(m):>8} | {md:8.2f} {1e3/md:7.1f} "
              f"{(d.max()-d.min())/md*100:7.1f}% {b_d/md:11.3f}x | "
              f"{mp:9.1f} {PREFILL_T/mp*1e3:7.1f} {b_p/mp:7.3f}x")
    print(f"\nсвоп: {sw0:.0f} -> {sw1:.0f} МБ "
          f"({'ОК' if sw1 - sw0 < 1 else 'ЗАМЕР НЕДЕЙСТВИТЕЛЕН (закон 11)'})")


if __name__ == "__main__":
    main()
