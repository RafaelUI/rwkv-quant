"""
Скорость нашего кернеля против llama.cpp на одной машине и одной модели.

Меряется то же, что llama-bench:
  pp512 -- префилл 512 токенов (throughput, GEMM-bound)
  tg128 -- генерация 128 токенов со state (decode, memory-bound)

Пары для сравнения по размеру:
  COMPRESSION (970 МБ)  <-> Q4_K_M (990 МБ)
  REDUCTION   (1255 МБ) <-> Q6_K   (1336 МБ)

Запуск: python tests/bench_decode_vs_gguf.py > /tmp/bench.log 2>&1 &
"""
import copy
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mlx.core as mx  # noqa: E402

from rwkv_quant.presets import REDUCTION, COMPRESSION  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
ACT_STATS = "/tmp/act_stats_1p5b_multiling.pt"
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536
PP, TG, WARMUP = 512, 128, 8


def cfg_of(preset):
    c = copy.deepcopy(preset)
    c.act_stats_path = ACT_STATS
    c.bits["small"] = 16          # подтверждено на обоих масштабах
    return c


def swap_mb():
    """vm.swapusage через LC_ALL=C: локаль с запятичным разделителем иначе
    ломает разбор."""
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def bench(model):
    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, PP)).astype(np.int32))

    # Префилл: полный прогон PP токенов с нуля, через СКОМПИЛИРОВАННЫЙ
    # путь. До 15.08 здесь стоял сырой forward_stateful, хотя декод рядом
    # уже шёл через model.step -- то есть половина таблицы сравнивалась с
    # llama.cpp на компилированном пути, а половина на сыром. Цена ошибки:
    # 344 против 533 ток/с, то есть "префилл проигрываем вдвое"
    # превращается в 1.3x (tests/bench_compile_ab.py).
    st = model.init_state(1)
    logits, st = model.step(prompt, st, True)
    mx.eval(logits)               # первый вызов = трассировка на эту форму
    st = model.init_state(1)
    t0 = time.time()
    logits, st = model.step(prompt, st, True)
    mx.eval(logits)
    pp = PP / (time.time() - t0)

    # Своп фиксируется на ГРАНИЦЕ измерения, а не от старта процесса:
    # сборка модели из .pth держит воркспейс грид-поиска и законно
    # выталкивает чужие страницы. Недействительным замер делает пейджинг
    # ВО ВРЕМЯ измерения (закон 11 и уточнение к нему в bench_sym_e2e_ab).
    # Прежде этот скрипт своп не смотрел вовсе, из-за чего прогон 15.08
    # пришлось отбросить задним числом и по внешним признакам.
    sw0 = swap_mb()

    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(WARMUP):       # прогрев кеша T=1
        logits, st = model.step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok)
    t0 = time.time()
    for _ in range(TG):
        logits, st = model.step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok)
    dt = time.time() - t0
    return pp, TG / dt, dt / TG * 1000, sw0, swap_mb()


def main():
    sd = torch.load(CKPT_PTH, map_location="cpu", mmap=True)
    for name, preset in (("COMPRESSION", COMPRESSION), ("REDUCTION", REDUCTION)):
        cfg = cfg_of(preset)
        t0 = time.time()
        ckpt = QuantizedCheckpoint(
            naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD, head_size=HEAD_SIZE,
            vocab_size=VOCAB, config_repr=repr(cfg),
            tensors={k: quantize_tensor(k, w, cfg, real_gw=True)
                     for k, w in sd.items()})
        model = QuantRWKV7(ckpt)
        del ckpt
        gc.collect()
        pp, tg, ms, sw0, sw1 = bench(model)
        verdict = ("своп не рос" if sw1 <= sw0 else
                   f"СВОП ВЫРОС на {sw1-sw0:.0f} МБ -- ЗАМЕР НЕДЕЙСТВИТЕЛЕН")
        print(f"{name:<14} pp{PP}={pp:7.1f} t/s   tg{TG}={tg:6.2f} t/s "
              f"({ms:.2f} мс/ток)   [сборка {time.time()-t0:.0f}s]", flush=True)
        print(f"{'':<14} своп {sw0:.0f} -> {sw1:.0f} МБ: {verdict}", flush=True)
        del model
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
