"""Разбивка трафика декода по статьям: линейные, LoRA, мелочь.

Счётчик после правки 09.09 даёт 946.2 МБ, README обещает 878. Разница в 8%
переезжает прямо в ГБ/с, поэтому надо видеть слагаемые, а не итог.
"""
import os
import sys

sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")

import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as _qm
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.formats.reader import load_raw
from trace_decode_steady import decode_traffic_mb

PATH = sys.argv[1]


def main():
    m = QuantRWKV7(load_raw(PATH))
    keep, seen = [], set()
    def sz(objs):
        t = 0
        for a in objs:
            if isinstance(a, mx.array) and id(a) not in seen:
                seen.add(id(a))
                keep.append(a)
                t += a.nbytes
        return t / 1e6
    head = sz(vars(m.head).values())
    proj = cmixk = cmixv = lora_q = lora_d = small = 0.0
    for b in m.blocks:
        tm, cm = b.tmix, b.cmix
        for l in (tm.r_proj, tm.k_proj, tm.v_proj, tm.o_proj):
            proj += sz(vars(l).values())
        cmixk += sz(vars(cm.key).values())
        cmixv += sz(vars(cm.value).values())
        if getattr(tm, "_lq_A", None) is not None:
            for tr in (tm._lq_A + tm._lq_B):
                lora_q += sz(list(tr))
        lora_d += sz([tm.w_lora_A, tm.w_lora_B_w, tm.a_lora_A, tm.a_lora_B_w,
                      tm.v_lora_A, tm.v_lora_B_w, tm.g_lora_A, tm.g_lora_B_w])
        small += sz([tm.k_k, tm.k_a, tm.r_k, tm.x_r, tm.x_w, tm.x_k, tm.x_v,
                     tm.x_a, tm.x_g, tm.ln_x_w, tm.ln_x_b, cm.x_k,
                     b.ln1_w, b.ln1_b, b.ln2_w, b.ln2_b])
    print("файл на диске        %8.1f МБ" % (os.path.getsize(PATH) / 1e6), flush=True)
    print("LORA_Q = %r" % (_qm.LORA_Q,), flush=True)
    print("", flush=True)
    print("статья                    МБ", flush=True)
    print("head                 %8.1f" % head, flush=True)
    print("proj r/k/v/o x24     %8.1f" % proj, flush=True)
    print("cmix.key x24         %8.1f" % cmixk, flush=True)
    print("cmix.value x24       %8.1f" % cmixv, flush=True)
    print("LoRA квантованная    %8.1f  (АКТИВНА при sep)" % lora_q, flush=True)
    print("LoRA плотная         %8.1f  (не читается при sep)" % lora_d, flush=True)
    print("мелочь (нормы/лерпы) %8.1f" % small, flush=True)
    print("", flush=True)
    print("ЧИТАЕТСЯ ЗА ТОКЕН    %8.1f  (без плотной LoRA)"
          % (head + proj + cmixk + cmixv + lora_q + small), flush=True)
    print("штатный счётчик      %8.1f" % decode_traffic_mb(m), flush=True)


if __name__ == "__main__":
    main()
