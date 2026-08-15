"""РОВНАЯ декодная нагрузка для Metal System Trace.

Зачем отдельный скрипт, а не бенч. Бенчи чередуют конфиги и меряют
медианы -- для трассы это худшее, что можно сделать: окно захвата
короткое (15 с), и если внутри него нагрузка меняется, счётчики
показывают смесь. Здесь наоборот: ОДИН конфиг, одна и та же операция на
каждом шаге, никаких периодических всплесков, и фаза держится минуту с
лишним, чтобы окно можно было взять где угодно в середине.

Что печатается по ходу -- ms/ток за последние 5 с и своп. Печать раз в
пять секунд на фоне ~270 шагов пренебрежима, но если понадобится совсем
чистое окно -- SILENT=1.

    python tests/trace_decode_steady.py reduction_sym_head8 [секунды]
    SILENT=1 python tests/trace_decode_steady.py reduction 90
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402

import ablate_sym_composite as comp  # noqa: E402

SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
SILENT = os.environ.get("SILENT") == "1"
WARMUP = 32


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


def main():
    name = sys.argv[1]
    cfg = comp.CONFIGS[name]()
    print(f"PID {os.getpid()}   конфиг {name}   чекпоинт "
          f"{os.path.basename(comp.CKPT)}", flush=True)

    sd = torch.load(comp.CKPT, map_location="cpu", mmap=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    emb = sd["emb.weight"]
    r_k = next(v for k, v in sd.items() if k.endswith("r_k"))
    meta = dict(naming="world", n_layer=n_layer, n_embd=int(emb.shape[1]),
                vocab_size=int(emb.shape[0]), head_size=int(r_k.shape[-1]))
    tensors = {k: quantize_tensor(k, w, cfg, real_gw=True) for k, w in sd.items()}
    model = QuantRWKV7(QuantizedCheckpoint(tensors=tensors, config_repr=repr(cfg),
                                           **meta))
    del tensors, sd
    gc.collect()
    mx.clear_cache()

    st = model.init_state(1)
    tok = mx.array(np.array([1], dtype=np.int32))
    for _ in range(WARMUP):                       # прогрев mx.compile
        logits, st = model.step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok)

    print(f"\n>>> УСТОЙЧИВАЯ ФАЗА НАЧАЛАСЬ, {SECONDS:.0f} с.", flush=True)
    print(">>> ЗАХВАТ БРАТЬ В ПЕРВЫЕ 20-25 СЕКУНД. Замерено на M4 air: "
          "18.1 мс/ток\n>>> первые 15 с и 22-26 мс/ток к шестидесятой -- "
          "безвентиляторный троттлинг\n>>> −30..40%. Окно, взятое позже, "
          "описывает перегретую машину, а не кернель.", flush=True)
    print(f">>> своп на входе {swap_mb():.1f} МБ — если он вырастет, "
          f"трасса недействительна (закон 11)\n", flush=True)

    t_start = time.perf_counter()
    t_mark, n_mark, n_total, first_ms = t_start, 0, 0, None
    while True:
        logits, st = model.step(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok)
        n_mark += 1
        n_total += 1
        now = time.perf_counter()
        if now - t_mark >= 5.0:
            ms = 1000 * (now - t_mark) / n_mark
            if first_ms is None:
                first_ms = ms
            drift = 100 * (ms - first_ms) / first_ms
            if not SILENT:
                print(f"  [{now-t_start:5.1f} с] {ms:6.2f} мс/ток   "
                      f"{n_mark/(now-t_mark):5.1f} ток/с   "
                      f"своп {swap_mb():.1f} МБ   дрейф {drift:+5.1f}%"
                      + ("   <-- ТРОТТЛИНГ, окно отсюда уже не про кернель"
                         if drift > 10 else ""), flush=True)
            t_mark, n_mark = now, 0
        if now - t_start >= SECONDS:
            break
    dt = time.perf_counter() - t_start
    print(f"\n<<< ФАЗА ЗАКОНЧЕНА: {n_total} токенов за {dt:.1f} с = "
          f"{1000*dt/n_total:.2f} мс/ток, {n_total/dt:.1f} ток/с")
    print(f"<<< своп на выходе {swap_mb():.1f} МБ")


if __name__ == "__main__":
    main()
