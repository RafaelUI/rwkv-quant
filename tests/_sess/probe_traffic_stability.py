"""Устойчивость decode_traffic_mb: тот же файл, тот же процесс, пять вызовов.

09.09: два прогона bench_gemv_presets_ab на ОДНИХ файлах дали у
compression_v2_cand трафик 851.1 и 913.9 МБ. Гипотеза: dedup идёт по id()
объекта, который НЕ удерживается -- временный mx.array освобождается сразу
после add(), его id достаётся следующему тензору, и следующий объявляется
уже посчитанным. Тогда недосчёт зависит от аллокатора, то есть плавает.
"""
import gc
import os
import sys

sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")

import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as _qm
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.formats.reader import load_raw
from trace_decode_steady import decode_traffic_mb

FIELDS = ("qblk", "qs", "d", "codes", "scales", "biases", "w", "mlx_weight")
PATH = sys.argv[1]


def walk(model, retain):
    seen, keep, tot, n, dup = set(), [], 0, 0, 0
    def add(a):
        nonlocal tot, n, dup
        if isinstance(a, mx.array):
            if id(a) in seen:
                dup += 1
                return
            seen.add(id(a))
            if retain:
                keep.append(a)
            tot += a.nbytes
            n += 1
    def lin(l):
        for f in FIELDS:
            add(getattr(l, f, None))
    lin(model.head)
    for b in model.blocks:
        tm, cm = b.tmix, b.cmix
        for l in (tm.r_proj, tm.k_proj, tm.v_proj, tm.o_proj, cm.key, cm.value):
            lin(l)
        if _qm.LORA_Q and getattr(tm, "_lq_A", None) is not None:
            for tr in (tm._lq_A + tm._lq_B):
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
    return tot / 1e6, n, dup


def main():
    m = QuantRWKV7(load_raw(PATH))
    print("файл: " + os.path.basename(PATH), flush=True)
    print("LORA_Q = " + repr(_qm.LORA_Q), flush=True)
    l = m.blocks[0].tmix.r_proj
    print("отдаёт ли доступ СВЕЖИЙ объект (r_proj слоя 0):", flush=True)
    for f in FIELDS:
        x = getattr(l, f, None)
        if isinstance(x, mx.array):
            y = getattr(l, f, None)
            same = "тот же" if id(x) == id(y) else "СВЕЖИЙ КАЖДЫЙ РАЗ"
            print("  %-11s %-18s %8.2f МБ" % (f, same, x.nbytes / 1e6), flush=True)
    for i in range(5):
        a = decode_traffic_mb(m)
        b, nb, db = walk(m, False)
        c, nc, dc = walk(m, True)
        print("вызов %d: штатный %8.1f | без удержания %8.1f (n=%d dedup=%d)"
              " | С УДЕРЖАНИЕМ %8.1f (n=%d dedup=%d)"
              % (i + 1, a, b, nb, db, c, nc, dc), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
