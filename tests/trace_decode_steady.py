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

Аргументом можно дать И имя конфига (тогда модель собирается из .pth,
~30 с), И ГОТОВЫЙ `.rwkvq` -- второе честнее, если наблюдают за тем, что
реально отгружается, и быстрее на полминуты.

ТРАФИК ЗА ТОКЕН ПЕЧАТАЕТСЯ ОБХОДОМ ЖИВЫХ БУФЕРОВ, а не формулой: только
то, что путь декода реально читает. `emb` исключён -- это gather одной
строки, а не таблица (закон 12); из двух копий LoRA считается ТА, что
активна при нынешнем `LORA_Q`. Это даёт число, прямо сравнимое с
показаниями внешнего профилировщика полосы: он видит ВЕСЬ трафик, то есть
должен показать чуть больше нашего (активации, состояние WKV, накладные),
и если он показывает МЕНЬШЕ -- значит часть чтений гасится кэшем и
наша арифметика полосы завышена.

    python tests/trace_decode_steady.py /tmp/reduction_new.rwkvq [секунды]
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


def decode_traffic_mb(model):
    """Байты, которые путь ДЕКОДА читает за один токен.

    Обход живых буферов, а не формула бит/вес: формула уже устаревала
    молча (в bench_molly_ab она считала LoRA плотным fp16 у обеих
    моделей и разъехалась с реальностью в тот день, когда LoRA
    квантовали). `emb` исключается -- gather одной строки."""
    import rwkv_quant.backends.metal.quant_model as _qm
    seen, tot = set(), 0

    def add(a):
        nonlocal tot
        if isinstance(a, mx.array) and id(a) not in seen:
            seen.add(id(a))
            tot += a.nbytes

    def lin_bytes(l):
        for f in ("qblk", "qs", "d", "codes", "scales", "biases", "w",
                  "mlx_weight"):
            add(getattr(l, f, None))

    lin_bytes(model.head)
    for b in model.blocks:
        tm, cm = b.tmix, b.cmix
        for l in (tm.r_proj, tm.k_proj, tm.v_proj, tm.o_proj, cm.key, cm.value):
            lin_bytes(l)
        if _qm.LORA_Q and getattr(tm, "_lq_A", None) is not None:
            for tr in (tm._lq_A + tm._lq_B):     # активны квантованные
                for a in tr:
                    add(a)
        else:
            for a in (tm.w_lora_A, tm.w_lora_B_w, tm.a_lora_A, tm.a_lora_B_w,
                      tm.v_lora_A, tm.v_lora_B_w, tm.g_lora_A, tm.g_lora_B_w):
                add(a)
        for a in (tm.k_k, tm.k_a, tm.r_k, tm.x_r, tm.x_w, tm.x_k, tm.x_v,
                  tm.x_a, tm.x_g, tm.ln_x_w, tm.ln_x_b, cm.x_k,
                  b.ln1_w, b.ln1_b, b.ln2_w, b.ln2_b):
            add(a)
    return tot / 1e6


def main():
    name = sys.argv[1]
    if name.endswith(".rwkvq"):
        from rwkv_quant.formats.reader import load_raw
        print(f"PID {os.getpid()}   файл {os.path.basename(name)}", flush=True)
        model = QuantRWKV7(load_raw(name))
        return run(model, name)
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
    return run(model, name)


def run(model, name):
    import rwkv_quant.backends.metal.quant_model as _qm
    traf = decode_traffic_mb(model)
    print(f"трафик за токен (обход живых буферов, без emb): {traf:.1f} МБ | "
          f"FUSE={_qm.FUSE} LORA_Q={_qm.LORA_Q}@{_qm.LORA_QBITS}", flush=True)
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
                      f"{traf/ms:5.1f} ГБ/с   "
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
