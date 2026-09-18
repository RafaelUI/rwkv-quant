"""Байты за токен декода по статьям (GEMV / LoRA) для перевода статей
bench_decode_rest_ab в ГБ/с. Счёт ПОСЛЕ шага декода (дефект 3 от 10.09),
обход vars() с удержанием ссылок (дефекты 1-2)."""
import sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import mlx.core as mx
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.formats.reader import load_raw
m = qm.QuantRWKV7(load_raw("/Users/s/Develop/WKV-kvant/" + sys.argv[1]))
lg, _ = m.forward_stateful(mx.array([[1]], dtype=mx.int32), m.init_state(1)); mx.eval(lg)
seen, keep = set(), []
def nb(objs):
    t = 0
    for a in objs:
        if isinstance(a, mx.array) and id(a) not in seen:
            seen.add(id(a)); keep.append(a); t += a.nbytes
    return t
g = nb(vars(m.head).values()); head = g
for b in m.blocks:
    for l in (b.tmix.r_proj, b.tmix.k_proj, b.tmix.v_proj, b.tmix.o_proj, b.cmix.key, b.cmix.value):
        g += nb(vars(l).values())
lo = 0
for b in m.blocks:
    tm = b.tmix
    lo += nb([a for tr in (tm._lq_A + tm._lq_B) for a in tr])
print("%s GEMV %.1f МБ (голова %.1f), LoRA %.1f МБ" % (sys.argv[1], g/1e6, head/1e6, lo/1e6))
