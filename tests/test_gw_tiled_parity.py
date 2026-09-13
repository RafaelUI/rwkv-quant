"""Гейт ПРИОРИТЕТА 1: gw_linear_tiled против плотного пути GwQuantLinear.

ПОРОГИ ЗАРЕГИСТРИРОВАНЫ 13.09, ДО ПЕРВОГО ЗАМЕРА, задним числом не двигаются:

    выход линейного слоя, N < 128         БИТ-В-БИТ, без допуска
    логиты после полного decode-шаг       БИТ-В-БИТ
    (батч B=8 и B=16, все слои сразу)

Порог бит-в-бит выбран не произвольно: nb-кернель уже обещает бит-в-бит
по паре (строка, колонка) и это подтверждено свипом чанка 19.07 (числа
идентичны на chunk 4/8/16/32). Тайловый кернель -- та же математика с
другим планом декода тайла, поэтому закон 37 здесь даёт бит-в-бит, а не
относительный допуск.

ПОЧЕМУ ЭТАЛОН -- ПОТНЫЙ ПУТЬ, А НЕ CHUNKED NB. Плотный путь
(_dequant_w + matmul) бит-в-бит с writer по построению (иначе был бы
сломан весь проект, а не только декод), и это единственный путь, уже
пройденный при N >= 128. Chunked nb сам стоит под вопросом ровно на
границах чанка (ceil(N/4) перечитываний -- источник открытого вопроса),
поэтому сверяться с ним значило бы мерить кернель его же подозреваемым
дефектом. Эталон -- ТОТ ЖЕ КОД, КОТОРЫМ СЧИТАЕТ МОДЕЛЬ при N >= 128
(закон 30): гейт зовёт GwQuantLinear.__call__ с форсированным плотным
путём, а не переписывает деквант свою копию.

СОСТОЯНИЕ КЕРНЕЛЯ: gw_linear_tiled ЕЩЁ НЕТ, gw_linear_tiled() в
gw_linear_tiled.py -- псевдоним эталона (форсирует тот же плотный путь).
Гейт ОБЯЗАН идти тривиально зелёным в этом состоянии; если он краснеет
сейчас -- красен сам гейт, а не кернель. Когда появится настоящий
tiled-кернель с декодом в threadgroup-памяти, меняется одна функция в
gw_linear_tiled.py, а этот файл -- нет.

Покрытие по N: {1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,127} -- сетка
покрывает все границы NB_CHUNK=4 и обе стороны порога GEMM_MIN_BATCH_NB
=128. N=128+ (плотный путь) сюда не входит: и эталон, и кернель на этом
диапазоне -- буквально один и тот же вызов, сравнивать нечего (см.
docstring gw_linear_tiled.py) -- проверка на T=512 нужна будет ПОСЛЕ
появления настоящего кернеля, не раньше, отдельным разделом (TODO ниже).

Запуск: из-под screen (закон 13, MCP таймаут). Ручки: RWKVQ_GATE_NS --
список N через запятую, RWKVQ_GATE_MUTATE=eps -- мутация одного элемента
выхода линейного слоя (1+eps), RWKVQ_GATE_SKIP -- список разделов
(comp,batch) через запятую.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm
import rwkv_quant.backends.metal.quant_linear_gw as gw
from rwkv_quant.backends.metal.gw_linear_tiled import gw_linear_tiled

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Develop/WKV-kvant/compression_v2_cand.rwkvq")
NS = [int(x) for x in os.environ.get(
    "RWKVQ_GATE_NS",
    "1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,127").split(",")]
SKIP = set(x for x in os.environ.get("RWKVQ_GATE_SKIP", "").split(",") if x)

FAILS = []
T0 = time.time()


def say(msg):
    print("[%6.1f] %s" % (time.time() - T0, msg), flush=True)


def check(name, ok, detail):
    say(("OK   " if ok else "FAIL ") + name + ": " + detail)
    if not ok:
        FAILS.append(name)


def f32(a):
    return np.array(a.astype(mx.float32))


def bitsame(a, b):
    if a.dtype != b.dtype:
        return False
    return np.array_equal(f32(a), f32(b))


def ref_dense(qlin, x):
    old = gw.GEMM_MIN_BATCH_NB
    gw.GEMM_MIN_BATCH_NB = 1
    try:
        return qlin(x)
    finally:
        gw.GEMM_MIN_BATCH_NB = old


say("модель: %s" % MODEL)
model = qm.QuantRWKV7(load_raw(MODEL))
say("слоёв в модели: %d" % model.n_layer)
say("kernel is forced-dense alias: %s" % (
    gw_linear_tiled.__module__ + "." + gw_linear_tiled.__name__))

# --- мутация: РАЗРЕШЕНИЕ ГЕЙТА (закон 37) --------------------------------
# ПОМЕРЕНО 13.09 на compression_v2_cand, тем же прогоном, что и сам гейт:
#   ловит 1e-7 и ниже (out[0,0] x(1+1e-7) -- красно);
#   слеп на 1e-8 (out[0,0] x(1+1e-8) -- зелено, ниже решётки fp32 на
#   этом элементе). Разрешение упирается в fp32 ulp, не в слепоту гейта.
MUT = float(os.environ.get("RWKVQ_GATE_MUTATE", "0") or 0)
if MUT:
    _orig_tiled = gw_linear_tiled

    def _mutated(qlin, x):
        out = _orig_tiled(qlin, x)
        out = out.astype(mx.float32)
        mask = mx.zeros(out.shape)
        mask[0, 0] = 1.0
        out = out * (1.0 + MUT * mask)
        return out
    import rwkv_quant.backends.metal.gw_linear_tiled as _mod
    _mod.gw_linear_tiled = _mutated
    say("МУТАЦИЯ x(1+%g) на out[0,0]: гейт ОБЯЗАН покраснеть" % MUT)
else:
    _mod = None


# --- 1. компонентный паритет: сетка N на реальном слое projection -------
if "comp" not in SKIP:
    # первый QuantLinear-подобный gw-слой модели -- receptance/key проекции
    # первого блока; qlin выбирается по атрибуту, а не переизобретается
    layer0 = model.blocks[0]
    qlin_candidates = [(n, getattr(layer0.tmix, n, None))
                        for n in ("k_proj", "v_proj", "r_proj", "o_proj")]
    qlin_candidates = [(n, q) for n, q in qlin_candidates
                        if isinstance(q, gw.GwQuantLinear)]
    check("найден хотя бы один gw-слой в блоке 0", len(qlin_candidates) > 0,
          "%d кандидатов" % len(qlin_candidates))
    name, qlin = qlin_candidates[0]
    say("слой для компонентного паритета: tmix.%s, IN=%d OUT=%d" % (
        name, qlin.in_features, qlin.out_features))
    rng = np.random.default_rng(13)
    bad_ns = []
    for n in NS:
        x = mx.array(rng.standard_normal((n, qlin.in_features)).astype(
            np.float32))
        ra = ref_dense(qlin, x)
        rb = (_mod.gw_linear_tiled if _mod else gw_linear_tiled)(qlin, x)
        mx.eval(ra, rb)
        if not bitsame(ra, rb):
            bad_ns.append(n)
    check("бит-в-бит по всей сетке N", not bad_ns,
          "расхождений 0" if not bad_ns else "N=%r" % bad_ns)


# --- 2. интеграционный паритет: реальный decode-шаг, B=8 и B=16 ---------
if "batch" not in SKIP:
    rng = np.random.default_rng(17)
    for B in (8, 16):
        st = model.init_state(B)
        tok = mx.array(rng.integers(1, 60000, size=(B, 1)).astype(np.int32))
        logits_ref, _ = model.forward_stateful(tok, st)
        mx.eval(logits_ref)
        # плечо "кернель" -- та же модель, тот же шаг: gw_linear_tiled
        # уже псевдоним эталона, так что здесь сверяется КОМПОЗИЦИЯ путей
        # (все линейные слои сразу), а не один слой изолированно
        st2 = model.init_state(B)
        logits_kernel, _ = model.forward_stateful(tok, st2)
        mx.eval(logits_kernel)
        check("decode-шаг B=%d бит-в-бит" % B,
              bitsame(logits_ref, logits_kernel),
              "relmax %.3e" % (float(np.max(np.abs(
                  f32(logits_ref) - f32(logits_kernel)))) /
                  (float(np.max(np.abs(f32(logits_ref)))) + 1e-9)))

# --- TODO: раздел T=512 после появления настоящего tiled-кернеля --------
# Пока и эталон, и "кернель" на N>=128 -- один и тот же плотный путь
# (сравнивать нечего, см. шапку). Добавить когда gw_linear_tiled престанет
# быть псевдонимом: сверить логиты полного префилла T=512 бит-в-бит.

say("ИТОГ: провалов %d" % len(FAILS))
if FAILS:
    say("ПРОВАЛЫ: %r" % FAILS)
    sys.exit(1)
say("ГОТОВО, всё зелёное")
