"""ГЕЙТ ДЛЯ ПОТРЕБИТЕЛЕЙ ФОРМАТА: sym-деквант в fp32 = codec, бит-в-бит.

Это гейт порта sym в rwkv-metal, и проверяет он ровно две вещи, которые
там могут разъехаться.

ПЕРВОЕ -- `SymQuantLinear.from_buffers`: конструктор из NUMPY-буферов,
которым и будет пользоваться QLoRA-база (torch у неё нет вовсе, файл она
читает через `codec.open_rwkvq`). Если интерлив собран не так, результат
будет ПРАВДОПОДОБНЫМ -- те же числа в другом порядке внутри пары блоков,
-- и ловится это только сверкой с эталоном.

ВТОРОЕ -- fp32-выход декванта. QLoRA восстанавливает базовый вес на
каждый forward, и он обязан совпадать с нормативным `codec.dequant_sym`
БИТ-В-БИТ: квантованная база -- отправная точка обучения, добавлять к ней
свой источник шума поверх калибровки нельзя. Право требовать равенства
есть: обе стороны считают `(q-32)*half(qs*d)` во float32.

КОНТРОЛЬ, БЕЗ КОТОРОГО ГЕЙТ БЫЛ БЫ ЗЕЛЁНЫМ ВСЕГДА: fp16-выход того же
кернеля с эталоном совпасть НЕ должен (half теряет биты). Если совпал --
значит сверяется не то, что думаем.

    python tests/test_sym_dequant_fp32.py <model.rwkvq> [ещё...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.formats import codec  # noqa: E402
from rwkv_quant.backends.metal.quant_linear_sym import SymQuantLinear  # noqa: E402

PATHS = [a for a in sys.argv[1:] if a.endswith(".rwkvq")] or \
    ["/tmp/reduction_new.rwkvq"]


def check(path):
    manifest, buf = codec.open_rwkvq(path)
    n_ok = n_bad = n_skip = 0
    bits_seen, amp, half_differs = set(), 0.0, 0
    for key, meta in manifest["tensors"].items():
        if meta.get("kind") != "sym":
            continue
        OUT, IN = tuple(meta["shape"])
        if IN % 256:
            n_skip += 1
            continue

        def b(field):
            return buf.get(f"{key}::{field}")

        lin = SymQuantLinear.from_buffers(
            shape=(OUT, IN), bits=meta["bits"], qs=b("gw_qs"), d=b("gw_d"),
            codes=b("codes"), codes_packed=b("codes_packed"),
            qh=b("gw_qh"), qh2=b("gw_qh2"))
        ref = codec.dequant_key(manifest, buf, key)
        got = np.asarray(lin._dequant_w(mx.float32))
        d = float(np.abs(got - ref).max())
        if d == 0.0:
            n_ok += 1
        else:
            n_bad += 1
            print(f"   РАСХОЖДЕНИЕ {key}: max|Δ| = {d:.3e}")
        # контроль: fp16 обязан отличаться, иначе гейт ничего не различает
        if float(np.abs(np.asarray(lin._dequant_w(mx.float16),
                                   dtype=np.float32) - ref).max()) > 0:
            half_differs += 1
        bits_seen.add(meta["bits"])
        amp = max(amp, float(np.abs(ref).max()))
        del lin, ref, got
        mx.clear_cache()
    return n_ok, n_bad, n_skip, bits_seen, amp, half_differs


total_bad = 0
for p in PATHS:
    ok, bad, skip, bits, amp, hd = check(p)
    total_bad += bad
    print(f"{os.path.basename(p)}: РАВНО {ok}, расхождений {bad}, "
          f"пропущено (IN не кратен 256) {skip}, битности {sorted(bits)}")
    print(f"   амплитуда эталона {amp:.3e}"
          f"{' -- ВЫРОЖДЕН' if amp < 1e-6 else ''}; "
          f"fp16 отличается от эталона на {hd} из {ok + bad} тензоров"
          f"{' -- ПОДОЗРИТЕЛЬНО, гейт может сверять не то' if hd == 0 else ''}")
    if ok == 0:
        total_bad += 1
        print("   НИ ОДНОГО sym-ТЕНЗОРА: гейт вырожден")

print("ГЕЙТ ЗЕЛЁНЫЙ" if total_bad == 0 else f"ГЕЙТ КРАСНЫЙ ({total_bad})")
sys.exit(1 if total_bad else 0)
