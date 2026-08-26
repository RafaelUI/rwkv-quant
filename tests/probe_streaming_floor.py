"""ИЗМЕРЕННЫЙ ПОЛ ДЛЯ ПОРОГА `test_streaming_inference` (закон 36).

Тест сравнивает ДВА РАЗНЫХ ЯДРА -- `wkv7_infer` (stateful) против
`wkv7_train` (батчевый путь) -- и порог 1e-4 на rel1 был НАЗНАЧЕН, а не
измерен. Два разных ядра обязаны разойтись на перестановке сумм, и вопрос
не в том, равны ли они, а в том, ДАЛЬШЕ ЛИ они друг от друга, чем две
заведомо эквивалентные редакции ОДНОЙ реализации.

Здесь строятся три пола, все -- одним и тем же инструментом (rel по
логитам на том же промпте):

  P1  ПЕРЕСТАНОВКА ПАДДИНГА: батчевый путь дополняет T=40 до кратного
      CHUNK. Смена CHUNK (16 -> 8/32) меняет и число no-op шагов, и
      раскладку чекпоинтов, оставляя величину МАТЕМАТИЧЕСКИ ТОЙ ЖЕ.
  P2  ПЕРЕСТАНОВКА ЧАНКОВАНИЯ У INFER: T=40 одним вызовом против 32+8 и
      20+20 с переносом state. Ядро одно, константа T разная.
  P3  ОТКЛИК НА ОДИН ULP fp32: выход WKV умножается на (1 + 2^-23), то
      есть на величину, неотличимую от самой себя в fp32. Это ВЕРХНЯЯ
      граница того, что даёт любая перестановка сумм внутри ядра, и она
      меряет УСИЛЕНИЕ модели от fp32-шума до логитов.

Печатает заодно решётку типа: логиты приходят через fp16-веса, и ниже
одного ulp fp16 расхождение представлено быть не может -- порог,
лежащий под решёткой, требует бит-в-бит равенства двух РАЗНЫХ ядер.

    python tests/probe_streaming_floor.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import mlx.core as mx

from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.writer import save as save_rwkvq
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal import quant_model as qm
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from tests.test_quant_model_smoke import build_toy_state_dict, N_LAYER, D, HEAD_SIZE, VOCAB

# ВНИМАНИЕ: `rwkv_metal.kernel.wkv7` -- это ФУНКЦИЯ, реэкспортированная
# в __init__ пакета, и она перекрывает одноимённый МОДУЛЬ при обычном
# импорте. Модуль берётся только через importlib/sys.modules.
import importlib
kw = importlib.import_module("rwkv_metal.kernel.wkv7")
# CHUNK живёт В ДВУХ модулях: `wkv7.CHUNK` решает, докуда паддить,
# `wkv7_checkpoint.CHUNK` зашивается в кернель и требует T % CHUNK == 0.
# Менять надо ОБА -- иначе сборка кернеля падает ассертом.
kwc = importlib.import_module("rwkv_metal.kernel.wkv7_checkpoint")

torch.manual_seed(0)
RWKVQ_PATH = "/tmp/toy_streaming.rwkvq"

TOKENS = [1, 5, 17, 42, 100, 3, 9, 200, 7, 88, 15, 33, 91, 4, 62, 19,
          23, 71, 8, 45, 2, 66, 12, 90, 34, 55, 21, 6, 77, 40,
          29, 11, 60, 3, 99, 18, 27, 5, 84, 50]


def rel(a, b, scale):
    return float(np.abs(a - b).max() / (scale + 1e-8))


def stream(model, idx, splits):
    """forward_stateful кусками длин splits с переносом state."""
    st = model.init_state(batch_size=1)
    out, p = [], 0
    for s in splits:
        lg, st = model.forward_stateful(idx[:, p:p + s], st)
        out.append(np.array(lg))
        p += s
    return np.concatenate(out, axis=1)


def main():
    sd = build_toy_state_dict()
    cfg = QuantConfig(proj=8, cmix=8, outlier_fracs={"proj": 0.02, "cmix": 0.01})
    save_rwkvq(sd, cfg, RWKVQ_PATH, naming="custom", n_layer=N_LAYER, n_embd=D,
               head_size=HEAD_SIZE, vocab_size=VOCAB)
    model = QuantRWKV7(load_raw(RWKVQ_PATH))
    idx = mx.array([TOKENS])

    ref_mx = model(idx)
    ref = np.array(ref_mx)
    scale = float(np.abs(ref).max())
    print("логиты: dtype %s, max|logits| = %.4f" % (ref_mx.dtype, scale))
    ulp16 = float(np.spacing(np.float16(scale)))
    print("решётка: один ulp fp16 на этой величине = %.3e (rel %.3e)"
          % (ulp16, ulp16 / scale))
    print("          один ulp fp32 = rel %.3e\n" % 2 ** -23)

    # ── ЧТО МЕРЯЕТ ТЕСТ ────────────────────────────────────────────────
    batched = stream(model, idx, [40])
    tok = stream(model, idx, [1] * 40)
    r1 = rel(ref, batched, scale)
    r2 = rel(ref, tok, scale)
    print("ИЗМЕРЯЕМОЕ (два РАЗНЫХ ядра):")
    print("  rel1 batched-stateful vs wkv7_train  %.3e  (%.2f ulp fp16)"
          % (r1, r1 * scale / ulp16))
    print("  rel2 token-by-token   vs wkv7_train  %.3e  (%.2f ulp fp16)\n"
          % (r2, r2 * scale / ulp16))

    # ── P1: перестановка паддинга у батчевого пути ─────────────────────
    print("ПОЛ P1 (та же реализация, другой CHUNK -- другой паддинг):")
    chunk0 = kw.CHUNK
    for c in (8, 32):
        def _set(v):
            kw.CHUNK = kwc.CHUNK = v
            kw._ckpt_cache.clear(); kw._ckpt_state_cache.clear()
            kwc._fwd_cache.clear(); kwc._bwd_cache.clear(); kwc._bwd2_cache.clear()
        try:
            _set(c)
            got = np.array(model(idx))
            print("  CHUNK %2d vs %d: rel %.3e  (паддинг %d -> %d)"
                  % (c, chunk0, rel(ref, got, scale),
                     -(-40 // chunk0) * chunk0, -(-40 // c) * c))
        finally:
            _set(chunk0)

    # ── P2: перестановка чанкования у infer ────────────────────────────
    print("\nПОЛ P2 (то же ядро infer, другое чанкование):")
    for splits in ([32, 8], [20, 20], [16, 16, 8]):
        got = stream(model, idx, splits)
        print("  %-12s vs [40]: rel %.3e" % (str(splits), rel(batched, got, scale)))

    # ── P3: отклик на один ulp fp32 в выходе WKV ───────────────────────
    print("\nПОЛ P3 (выход WKV домножен на 1+eps -- отклик логитов):")
    real = qm.wkv7_train
    for p in (23, 22, 20):
        eps = 2.0 ** -p
        try:
            qm.wkv7_train = (lambda *a, _e=eps, **kw_: real(*a, **kw_) * (1.0 + _e))
            got = np.array(model(idx))
        finally:
            qm.wkv7_train = real
        print("  eps 2^-%d = %.2e: rel %.3e" % (p, eps, rel(ref, got, scale)))


if __name__ == "__main__":
    main()
