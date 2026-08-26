"""МУТАЦИОННАЯ КРЫШКА ДЛЯ test_streaming_inference.

Порог поднят с 1e-4 до 1e-3 по ИЗМЕРЕННОМУ полу (probe_streaming_floor).
Поднятый порог обязан оставаться ловящим -- иначе вместо красного теста,
падавшего на арифметике, мы получим зелёный, не проверяющий ничего.

Две мутации, обе -- в stateful-путь (эталон `wkv7_train` не трогается):

  state   -- forward_stateful игнорирует переданный state и каждый раз
             начинает с нулевого. Это ровно та поломка, ради которой тест
             написан; она обязана убить плечо token-by-token и НЕ тронуть
             батчевое (там state и так стартует нулевым) -- то есть
             мутация проверяет ещё и что два плеча меряют РАЗНОЕ.
  kernel  -- выход wkv7_infer домножен на 1+2^-8, то есть ошибка 0.39%.
             ИЗМЕРЕНО: этот тест её НЕ ЛОВИТ и не должен -- голова считается
             в fp16, логиты ложатся на решётку 2^-11, и 0.39% на выходе WKV
             сдвигает их ровно на один ulp. Разрешения по величине у
             инструмента нет. Точность ядер живёт в отдельном гейте
             tests/test_wkv_infer_vs_train.py (fp64-эталон на CPU, та же
             мутация ловится с запасом 39x). Здесь она оставлена как
             ЗАПИСАННАЯ ГРАНИЦА ПРИМЕНИМОСТИ: если она вдруг начнёт
             ловиться, значит инструмент изменился и границу надо
             перемерить.

    python tests/mutate_streaming_gate.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mlx.core as mx

from rwkv_quant.backends.metal import quant_model as qm
import tests.test_streaming_inference as T


def run(label, expect_fail):
    ok = T.main()
    verdict = "поймана" if not ok else "НЕ ПОЙМАНА"
    if not expect_fail:
        verdict = "зелёный" if ok else "КРАСНЫЙ БЕЗ МУТАЦИИ"
    print("  -> %s: %s\n" % (label, verdict), flush=True)
    return (not ok) if expect_fail else ok


def main():
    good = []

    print("=== контроль: без мутации ===", flush=True)
    good.append(run("без мутации", expect_fail=False))

    print("=== мутация 'state': stateful-путь забывает состояние ===", flush=True)
    real_fs = qm.QuantRWKV7.forward_stateful
    try:
        def fs(self, idx, states, *a, **kw):
            return real_fs(self, idx, self.init_state(batch_size=idx.shape[0]),
                           *a, **kw)
        qm.QuantRWKV7.forward_stateful = fs
        good.append(run("state", expect_fail=True))
    finally:
        qm.QuantRWKV7.forward_stateful = real_fs

    print("=== мутация 'kernel': выход wkv7_infer x(1+2^-8) ===", flush=True)
    real_w = qm._wkv_stateful
    try:
        def w(*a, **kw):
            out, h = real_w(*a, **kw)
            return out * (1.0 + 2.0 ** -8), h
        qm._wkv_stateful = w
        ok_k = T.main()
        print("  -> kernel: %s (ожидание: НЕ ловится, см. шапку)\n"
              % ("не поймана, как и записано" if ok_k
                 else "ПОЙМАНА -- граница применимости сдвинулась"),
              flush=True)
        good.append(ok_k)
    finally:
        qm._wkv_stateful = real_w

    print("ИТОГ: %s" % ("крышка держит" if all(good)
                        else "КРЫШКА ДЫРЯВАЯ -- см. выше"))
    return all(good)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
