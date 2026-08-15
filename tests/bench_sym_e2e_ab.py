"""E2E-декод: пресет с Q6_K-раскладкой против нынешнего, ЧЕРЕДОВАНИЕМ.

Микро-A/B (bench_sym_vs_sb6_ab.py) меряет одну матрицу. Здесь -- цена
пресета целиком: у sym меняется кернель у cmix, proj и head, то есть у
88% файла, и вопрос стоит так -- сколько токенов в секунду стоит
падение бюджета с +0.772% до +0.192%.

ОБЕ МОДЕЛИ ЖИВУТ ОДНОВРЕМЕННО, и замеры чередуются (закон 1):
безвентиляторный M4 даёт дрейф до 1.8x между процессами, поэтому
«собрали одну, померили, собрали другую» -- не замер. Цена -- две
модели в памяти разом (~4.7 ГБ резидентно на 1.5B), поэтому своп
проверяется до и после: малейший рост делает замер недействительным.

    python tests/bench_sym_e2e_ab.py [ckpt]
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.formats.writer import TENSOR_FIELDS, quantize_tensor  # noqa: E402

import ablate_sym_composite as comp  # noqa: E402

CKPT = comp.CKPT
TG, WARMUP, ROUNDS = 64, 8, 5
VARIANTS = ["reduction", "reduction_sym_head8"]


def swap_mb():
    """Своп в МБ. LC_ALL=C ОБЯЗАТЕЛЕН: в локали с запятичным разделителем
    sysctl печатает "532,75M", и float() на этом падает. У автора скрипта
    оболочка была в C-локали, поэтому баг не воспроизводился -- классика
    «работает на моей машине». Запятая на всякий случай тоже разбирается."""
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]          # "532.75M" | "1.50G"
    unit, num = u[-1], u[:-1]
    if "," in num:                               # запятая -- десятичная
        num = num.replace(".", "").replace(",", ".")
    return float(num) * (1024 if unit == "G" else 1)


def build(name, sd, meta):
    cfg = comp.CONFIGS[name]()
    tensors = {k: quantize_tensor(k, w, cfg, real_gw=True)
               for k, w in sd.items()}
    mb = sum(sum(getattr(q, f).numel() * getattr(q, f).element_size()
                 for f in TENSOR_FIELDS if getattr(q, f, None) is not None)
             for q in tensors.values()) / 1e6
    m = QuantRWKV7(QuantizedCheckpoint(tensors=tensors, config_repr=repr(cfg),
                                       **meta))
    del tensors
    gc.collect()
    return m, mb


def decode_burst(model, tok, st, n):
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = model.step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok)
    return (time.perf_counter() - t0) / n * 1e3, tok, st


def main():
    sw0 = swap_mb()
    print(f"своп до: {sw0:.1f} МБ", flush=True)
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    emb = sd["emb.weight"]
    r_k = next(v for k, v in sd.items() if k.endswith("r_k"))
    meta = dict(naming="world", n_layer=n_layer, n_embd=int(emb.shape[1]),
                vocab_size=int(emb.shape[0]), head_size=int(r_k.shape[-1]))

    models, mbs = {}, {}
    for v in VARIANTS:
        t0 = time.time()
        models[v], mbs[v] = build(v, sd, meta)
        print(f"  {v}: {mbs[v]:.1f} МБ буферов, сборка {time.time()-t0:.0f}s",
              flush=True)
    del sd
    gc.collect()

    # Своп меряется НА ГРАНИЦЕ ЗАМЕРА, а не от старта процесса. Две
    # модели строятся из одного mmap-чекпоинта и на пике держат ещё и
    # воркспейс грид-поиска -- на 16 ГБ это законно выталкивает чужие
    # страницы, и своп растёт ДО первого бёрста. Недействительным замер
    # делает пейджинг ВО ВРЕМЯ измерения, а не факт, что система когда-то
    # свопила; поэтому фиксируется дельта между началом бёрстов и концом,
    # и она же печатается по раундам.
    mx.clear_cache()
    gc.collect()
    sw_bench = swap_mb()
    print(f"своп на входе в замер: {sw_bench:.1f} МБ "
          f"(за сборку {sw_bench - sw0:+.1f})", flush=True)

    state = {}
    for v in VARIANTS:
        st = models[v].init_state(1)
        tok = mx.array(np.array([1], dtype=np.int32))
        for _ in range(WARMUP):                      # прогрев mx.compile
            logits, st = models[v].step(tok[None], st)
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(tok)
        state[v] = (tok, st)

    acc = {v: [] for v in VARIANTS}
    for r in range(ROUNDS):
        for v in VARIANTS:                           # ЧЕРЕДОВАНИЕ
            tok, st = state[v]
            ms, tok, st = decode_burst(models[v], tok, st, TG)
            state[v] = (tok, st)
            acc[v].append(ms)
        print(f"  раунд {r+1}/{ROUNDS}: "
              + "  ".join(f"{v} {acc[v][-1]:.2f} мс" for v in VARIANTS)
              + f"  | своп {swap_mb():.1f} МБ", flush=True)

    sw1 = swap_mb()
    print(f"\nсвоп: старт {sw0:.1f} -> вход в замер {sw_bench:.1f} -> "
          f"конец {sw1:.1f} МБ")
    if sw1 > sw_bench + 0.5:
        print("*** ЗАМЕР НЕДЕЙСТВИТЕЛЕН: своп рос ВО ВРЕМЯ бёрстов "
              "(закон 11) ***")
        return 1
    print("своп во время бёрстов не рос -- замер действителен")

    print(f"\n{'вариант':22s} {'мс/ток':>8s} {'ток/с':>8s} {'МБ':>8s} "
          f"{'разброс':>8s}")
    base = None
    for v in VARIANTS:
        med = float(np.median(acc[v]))
        spread = 100 * (max(acc[v]) - min(acc[v])) / med
        base = med if base is None else base
        print(f"{v:22s} {med:8.2f} {1000/med:8.2f} {mbs[v]:8.1f} "
              f"{spread:7.1f}%")
    a, b = float(np.median(acc[VARIANTS[0]])), float(np.median(acc[VARIANTS[1]]))
    print(f"\n{VARIANTS[1]} против {VARIANTS[0]}: {100*(b-a)/a:+.1f}% по "
          f"времени при {100*(mbs[VARIANTS[1]]-mbs[VARIANTS[0]])/mbs[VARIANTS[0]]:+.1f}% "
          f"по размеру")
    print("если проценты совпали -- кернель sym не хуже, вся цена в байтах")
    return 0


if __name__ == "__main__":
    sys.exit(main())
