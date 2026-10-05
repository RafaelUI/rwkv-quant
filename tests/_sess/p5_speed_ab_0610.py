"""06.10, п. 5: цена выключения квантованных LoRA на декоде для reduction -- скорость generate (конвейер по умолчанию).
Две модели одного файла в одном процессе (lora_q=True / False: флаг читается при трассировке mx.compile, переключать его
на живой модели нельзя), чередование ABAB по раундам (машина без вентилятора: дрейф температуры общий для плеч).
    python p5_speed_ab_0610.py <файл.rwkvq> [раундов=8] [токенов=128]"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.backends.metal import generate as G
from rwkv_quant.formats.reader import load_raw
F = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 8; N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
swap = lambda: subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used = ")[1].split()[0]
ids = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt"), weights_only=False)["tokens"][12, :64].tolist()
s0 = swap(); assert qm.LORA_Q == "auto"
M = {"on": qm.QuantRWKV7(load_raw(F), lora_q=True), "off": qm.QuantRWKV7(load_raw(F), lora_q=False)}
toks = {}
for n, m in M.items(): toks[n], _ = G.generate(m, ids, N)          # прогрев и трассировка
built = {n: sum(getattr(b.tmix, "_lq_A", None) is not None for b in m.blocks) for n, m in M.items()}
assert built["on"] == len(M["on"].blocks) and built["off"] == 0, built
print("%s: пресет %s, намерение файла lora_q=%s (%s); жадные токены on == off: %s (%d из %d совпали с начала)" % (
    os.path.basename(F), M["on"].preset, qm.QuantRWKV7.__name__ and qm.resolve_lora_q(load_raw(F), None, M["on"].preset), "", toks["on"] == toks["off"],
    next((i for i, (a, b) in enumerate(zip(toks["on"], toks["off"])) if a != b), N), N), flush=True)
T = {"on": [], "off": []}
for r in range(R):
    for n in (("on", "off") if r % 2 == 0 else ("off", "on")):
        t0 = time.perf_counter(); G.generate(M[n], ids, N); T[n].append((time.perf_counter() - t0) / N * 1e3)
    print("  раунд %d: on %.2f мс/ток, off %.2f" % (r + 1, T["on"][-1], T["off"][-1]), flush=True)
a, b = np.array(T["on"]), np.array(T["off"]); d = b - a
print("ИТОГ %s: on (LoRA 8 бит) медиана %.2f мс/ток (%.1f ток/с), off (точные) %.2f (%.1f); off - on: медиана %+.2f мс (%+.1f%%), по раундам от %+.2f до %+.2f; своп %s -> %s"
      % (os.path.basename(F), np.median(a), 1e3 / np.median(a), np.median(b), 1e3 / np.median(b), np.median(d), 100 * np.median(d) / np.median(a), d.min(), d.max(), s0, swap()))
print("  (время generate включает префилл 64 токенов, одинаковый у плеч)")
