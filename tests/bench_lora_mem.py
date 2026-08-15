"""ПАМЯТЬ КВАНТОВАННЫХ LoRA-ВЕТОК: только постройка модели, без префилла.

Три состояния, и их НЕЛЬЗЯ путать:
  fp16   -- как сейчас: плотные fp16-копии, ~100 МБ на 1.5B;
  both   -- цена ВКЛЮЧЕНИЯ флага: обе копии живы, потому что A/B делается
            подменой флага в одном процессе (закон 27);
  qonly  -- цена ВНЕДРЕНИЯ: плотные освобождены (drop_lora_dense), в
            памяти только квантованные.

Пик снимается снаружи через /usr/bin/time -l (закон 22), поэтому один
режим на процесс. Гонять ЧЕРЕДОВАНИЕМ и на прогретом кеше страниц
(закон 26: первый прогон после записи файла меряет другое).

    /usr/bin/time -l python tests/bench_lora_mem.py <режим> [model.rwkvq]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "fp16"
PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/reduction_sym_head8.rwkvq"
BITS = int(os.environ.get("RWKVQ_LORA_BITS", 8))


def live_mb(model):
    """Прямой обход живых буферов -- не метрика фреймворка (закон 11)."""
    seen, tot = set(), 0

    def walk(o, d=0):
        nonlocal tot
        if d > 4:
            return
        for v in vars(o).values():
            for it in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(it, mx.array):
                    if id(it) not in seen:
                        seen.add(id(it))
                        tot += it.nbytes
                elif hasattr(it, "__dict__"):
                    walk(it, d + 1)
    walk(model)
    return tot / 1e6


def main():
    qm.LORA_QBITS = BITS
    qm.LORA_Q = None if MODE == "fp16" else "sep"
    model = qm.QuantRWKV7(load_raw(PATH))
    if MODE != "fp16":
        for b in model.blocks:
            b.tmix._build_lora_q()
        if MODE == "qonly":
            qm.drop_lora_dense(model)
        mx.eval([])
    print(f"{MODE}: живые буферы {live_mb(model):.1f} МБ "
          f"(LoRA {BITS} бит)", flush=True)


if __name__ == "__main__":
    main()
