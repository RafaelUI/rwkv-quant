"""Профиль decode-шага (T=1) на реальной 1.5B + REDUCTION.
Метод: полный tok/s, затем аблации (подмена компонента на дешёвую заглушку)
-- падение времени = доля компонента. Без eval-барьеров внутри графа.
Один процесс = один запуск (паттерн diagnose_one.py)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
import rwkv_quant.backends.metal.quant_model as qm

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

REDUCTION = QuantConfig(proj=8, cmix=8, emb_head=8, small=8)

def build():
    sd = torch.load(CKPT_PTH, map_location="cpu")
    tensors = {k: quantize_tensor(k, w, REDUCTION) for k, w in sd.items()}
    del sd
    ckpt = QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                               head_size=HEAD_SIZE, vocab_size=VOCAB,
                               tensors=tensors, config_repr="reduction")
    return qm.QuantRWKV7(ckpt)

def bench_decode(model, n_warm=5, n_iter=40):
    states = model.init_state(1)
    idx = mx.array(np.array([[123]], dtype=np.int64))
    # прогрев + вывод state в устоявшийся вид
    for _ in range(n_warm):
        logits, states = model.forward_stateful(idx, states)
        mx.eval(logits, *[s for st in states for s in st if s is not None])
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        logits, states = model.forward_stateful(idx, states)
        mx.eval(logits, *[s for st in states for s in st if s is not None])
    mx.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3

def main():
    print("building model (peak RSS будет тут)...", flush=True)
    t0 = time.perf_counter()
    model = build()
    print(f"built in {time.perf_counter()-t0:.1f}s", flush=True)

    # инвентаризация dense fp32
    import rwkv_quant.backends.metal.quant_linear as qlin
    import rwkv_quant.backends.metal.quant_linear_v2 as qlin2
    dense_bytes = 0
    quant_bytes = 0
    def sz(a): return a.size * a.itemsize if isinstance(a, mx.array) else 0
    for blk in model.blocks:
        for part in (blk.tmix, blk.cmix):
            for v in vars(part).values():
                if isinstance(v, mx.array):
                    dense_bytes += sz(v)
                elif isinstance(v, (qlin.QuantLinear, qlin2.QuantLinearV2)):
                    quant_bytes += v.codes.size * v.codes.itemsize + v.scale.size * 4
                elif hasattr(v, 'w') and isinstance(getattr(v, 'w', None), mx.array):
                    dense_bytes += sz(v.w)
    print(f"dense (fp32) в блоках: {dense_bytes/1e6:.1f} MB, quant codes: {quant_bytes/1e6:.1f} MB", flush=True)

    t_full = bench_decode(model)
    print(f"full step:            {t_full:6.2f} ms/tok  ({1000/t_full:.2f} tok/s)", flush=True)

    # -- аблация 1: WKV -> заглушка (возвращает v и state как есть)
    orig_wkv = qm._wkv_stateful
    qm._wkv_stateful = lambda r, w, k, v, a, b, state: (v, state)
    t_nowkv = bench_decode(model)
    qm._wkv_stateful = orig_wkv
    print(f"без WKV:              {t_nowkv:6.2f} ms/tok  (WKV ≈ {t_full-t_nowkv:.2f} ms)", flush=True)

    # -- аблация 2: head -> срез (только первые 16 строк)
    orig_head = model.head
    class TinyHead:
        def __call__(self, x): return x[..., :16]
    model.head = TinyHead()
    t_nohead = bench_decode(model)
    model.head = orig_head
    print(f"без head:             {t_nohead:6.2f} ms/tok  (head ≈ {t_full-t_nohead:.2f} ms)", flush=True)

    # -- аблация 3: LoRA-ветки g/a/w/v -> предвычисленные константы
    saved = []
    for blk in model.blocks:
        tm = blk.tmix
        saved.append((tm.g_lora_A, tm.a_lora_A, tm.w_lora_A,
                      getattr(tm, "v_lora_A", None)))
    # подменяем матрицы A на нулевые [1, D] -> матмулы схлопываются в мелочь
    D = N_EMBD
    for blk in model.blocks:
        tm = blk.tmix
        tm.g_lora_A = mx.zeros((1, D)); tm.g_lora_B_w = mx.zeros((D, 1))
        tm.a_lora_A = mx.zeros((1, D)); tm.a_lora_B_w = mx.zeros((D, 1))
        tm.w_lora_A = mx.zeros((1, D)); tm.w_lora_B_w = mx.zeros((D, 1))
        if getattr(tm, "v_lora_A", None) is not None:
            tm._v_saved = (tm.v_lora_A, tm.v_lora_B_w)
            tm.v_lora_A = mx.zeros((1, D)); tm.v_lora_B_w = mx.zeros((D, 1))
    t_nolora = bench_decode(model)
    print(f"LoRA rank-1:          {t_nolora:6.2f} ms/tok  (LoRA ≈ {t_full-t_nolora:.2f} ms)", flush=True)

    resid = t_nowkv + (t_full - t_nohead) + (t_full - t_nolora)
    print(f"\nостаток (матмулы+LN+обвязка) ≈ {t_full - (t_full-t_nowkv) - (t_full-t_nohead) - (t_full-t_nolora):.2f} ms", flush=True)

if __name__ == "__main__":
    main()
