"""
Проверка streaming-инференса (forward_stateful + wkv7_infer) на
игрушечном чекпоинте:

  1. forward_stateful(idx, init_state) на весь промпт одним вызовом ==
     __call__(idx) (wkv7_train, не-streaming путь) -- проверяет, что
     stateful-путь считает ту же математику, что батчевый.
     (No-op паддинга здесь больше нет: infer-кернель параметризован по T
     с кешем по (H, T) и принимает любую длину -- см. _wkv_stateful.)
  2. Тот же промпт, но token-by-token (T=1 за вызов, state переносится
     между вызовами) == тот же forward_stateful одним вызовом на весь
     промпт -- проверяет, что state корректно живёт между вызовами
     (это и есть весь смысл streaming decode).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import mlx.core as mx

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import save as save_rwkvq
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from tests.test_quant_model_smoke import build_toy_state_dict, N_LAYER, D, HEAD_SIZE, VOCAB

torch.manual_seed(0)

RWKVQ_PATH = "/tmp/toy_streaming.rwkvq"


def main():
    sd = build_toy_state_dict()
    cfg = QuantConfig(proj=8, cmix=8, outlier_fracs={"proj": 0.02, "cmix": 0.01})
    save_rwkvq(sd, cfg, RWKVQ_PATH, naming="custom", n_layer=N_LAYER, n_embd=D,
               head_size=HEAD_SIZE, vocab_size=VOCAB)
    ckpt = load_raw(RWKVQ_PATH)
    model = QuantRWKV7(ckpt)

    # T=40 нарочно (> CHUNK=32), чтобы задействовать и полный чанк, и хвост
    tokens = [1, 5, 17, 42, 100, 3, 9, 200, 7, 88, 15, 33, 91, 4, 62, 19,
              23, 71, 8, 45, 2, 66, 12, 90, 34, 55, 21, 6, 77, 40,
              29, 11, 60, 3, 99, 18, 27, 5, 84, 50]
    assert len(tokens) == 40
    idx_full = mx.array([tokens])

    # --- эталон: не-streaming forward (wkv7_train) ---
    logits_ref = np.array(model(idx_full))

    # --- 1) forward_stateful, весь промпт одним вызовом ---
    states0 = model.init_state(batch_size=1)
    logits_batched, _ = model.forward_stateful(idx_full, states0)
    logits_batched = np.array(logits_batched)

    err1 = np.abs(logits_ref - logits_batched).max()
    rel1 = err1 / (np.abs(logits_ref).max() + 1e-8)
    print(f"[batched-stateful vs wkv7_train]  max_abs_err={err1:.6f}  rel_err={rel1:.6e}")

    # --- 2) token-by-token, state переносится между вызовами ---
    states = model.init_state(batch_size=1)
    logits_stream = []
    for t in tokens:
        idx_t = mx.array([[t]])
        logits_t, states = model.forward_stateful(idx_t, states)
        logits_stream.append(np.array(logits_t)[0, 0])
    logits_stream = np.stack(logits_stream)[None, :, :]  # [1, T, VOCAB]

    err2 = np.abs(logits_ref - logits_stream).max()
    rel2 = err2 / (np.abs(logits_ref).max() + 1e-8)
    print(f"[token-by-token vs wkv7_train]    max_abs_err={err2:.6f}  rel_err={rel2:.6e}")

    top_ref = logits_ref[0, -1].argsort()[-5:]
    top_stream = logits_stream[0, -1].argsort()[-5:]
    overlap = len(set(top_ref.tolist()) & set(top_stream.tolist()))
    print(f"top-5 next-token overlap (last position): {overlap}/5")

    # ПОРОГ ИЗМЕРЕН, А НЕ НАЗНАЧЕН (закон 36; tests/probe_streaming_floor.py).
    #
    # Здесь сравниваются ДВА РАЗНЫХ ЯДРА -- wkv7_infer против wkv7_train --
    # и требовать от них бит-в-бит равенства нельзя: любые два порядка
    # суммирования обязаны разойтись. Вопрос в том, дальше ли они друг от
    # друга, чем две заведомо эквивалентные редакции ОДНОЙ реализации.
    #
    # Пол измерен тем же инструментом (rel по логитам, тот же промпт):
    #   * тот же infer-путь, но нарезанный иначе (32+8 и 16+16+8 против
    #     одного вызова на 40) -- rel 5.13e-04, то есть РОВНО один ulp fp16
    #     на масштабе логитов (max|logits| 0.9521, ulp 4.883e-04);
    #   * тот же train-путь при другом CHUNK (8/32 вместо 16, то есть
    #     другой паддинг и другая раскладка чекпоинтов) -- rel 0;
    #   * отклик логитов на домножение выхода WKV на 1+2^-20 -- 2.56e-04.
    # Измеряемые величины: rel1 = 2.564e-04 (ПОЛОВИНА ulp fp16, то есть
    # минимальное ненулевое значение, которое этот инструмент способен
    # показать), rel2 = 5.128e-04 (один ulp). ОБЕ не превосходят пола.
    #
    # Отсюда 1e-3 на оба плеча -- примерно два ulp fp16, вдвое выше
    # измеренного пола. Прежний порог 1e-4 на rel1 лежал ПОД решёткой
    # типа (0.5 ulp = 2.6e-4) и потому требовал бит-в-бит совпадения двух
    # разных ядер -- он падал не на баге, а на арифметике fp16.
    #
    # ПРЕЖНИЙ КОММЕНТАРИЙ УТВЕРЖДАЛ, ЧТО "расхождение батчевого пути
    # осталось нулевым". Это неверно и не могло быть верным: ноль там
    # означал бы, что два разных ядра совпадают побитово через 40 шагов
    # рекуррентности. Проверено на 1.5B REDUCTION, что это шум формата, а
    # не деградация: ppl fp16 против fp32 -- 11.5186 против 11.5191,
    # top-5 совпадает.
    # ГРАНИЦА ПРИМЕНИМОСТИ, ИЗМЕРЕННАЯ, А НЕ ПРЕДПОЛОЖЕННАЯ. Этот тест
    # проверяет ПЕРЕНОС STATE, и только его. По ВЕЛИЧИНЕ ошибки ядра он
    # слеп: мутация «выход wkv7_infer x(1+2^-8)», то есть 0.39%, сдвигает
    # логиты на один ulp fp16 и проходит здесь насквозь
    # (tests/mutate_streaming_gate.py). Точность ядер -- в
    # tests/test_wkv_infer_vs_train.py, где эталон считается в fp64 на CPU
    # и та же мутация ловится с запасом 39x. Прежний порог 1e-4 создавал
    # видимость проверки точности, а на деле требовал бит-в-бит.
    ok = rel1 < 1e-3 and rel2 < 1e-3
    print(f"\n[{'OK' if ok else 'FAIL'}] streaming inference: state carries correctly across calls")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
