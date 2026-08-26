"""ГЕЙТ: wkv7_infer против wkv7_train -- по ТОЧНОСТИ, а не по согласию.

ЗАЧЕМ ОТДЕЛЬНО ОТ test_streaming_inference. Тот сравнивает те же два ядра,
но ЧЕРЕЗ МОДЕЛЬ, а голова считается в fp16 -- логиты ложатся на решётку
2^-11, и всё, что мельче половины ulp, обращается в ноль. Измерено
мутацией (tests/mutate_streaming_gate.py): домножение выхода wkv7_infer на
1+2^-8, то есть ошибка в 0.39%, сдвигает логиты ровно на ОДИН ulp fp16 и
сквозь тот тест проходит. Разрешения по величине у него нет; его предмет
-- перенос state между вызовами, и только он.

ПОЧЕМУ НЕ «ДВА ЯДРА ДОЛЖНЫ СОВПАСТЬ С ТОЧНОСТЬЮ X». Такой порог пришлось
бы подобрать, а закон 36 запрещает: пол надо ИЗМЕРИТЬ эквивалентной
перестановкой той же реализации. Здесь перестановки нет -- рекуррентность
строго последовательна, и любая нарезка infer по времени выходит БИТ-В-БИТ
(проверено: нарезки T/2 и 1+(T-1) дают ноль). Значит эталона из этой семьи
не существует, и порог был бы фантазией.

ЧТО ДЕЛАЕТСЯ ВМЕСТО. Эталон считается в fp64 на CPU (numpy), и вопрос
ставится иначе: не «близки ли ядра друг к другу», а «одинаково ли они
далеки от точного значения». Критерий тогда БЕЗРАЗМЕРНЫЙ и ничего не
подбирает:

ГДЕ СТОИТ КРЫШКА, И ПОЧЕМУ ИМЕННО ТАМ. Оба конца ИЗМЕРЕНЫ:
  * шум -- ошибка обоих ядер против fp64 не превосходит 5.4e-06 на самой
    длинной боевой форме (T=512, H=40);
  * пойманная поломка -- мутация «выход x(1+2^-8)», то есть 0.39%, даёт
    3.9e-03.
Между ними три порядка, и крышка 1e-04 стоит примерно посередине в
логарифме: в 19 раз выше измеренного шума и в 39 раз ниже пойманной
ошибки. Это ЕДИНСТВЕННОЕ число в гейте, и оба его края -- замеры.

ЧЕГО ГЕЙТ НАРОЧНО НЕ ДЕЛАЕТ. Он не требует, чтобы ядра были одинаково
точны. Измерено: infer стабильно дальше от fp64, чем train, в 1.1-3.2
раза, и это свойство округления, а не дефект -- расхождение между ядрами
растёт как sqrt(T) (22.7x при sqrt(512) = 22.6, печатается ниже). Гейт по
такому отношению был бы красным на здоровом коде -- ровно та болезнь,
которую этот файл и лечит у test_streaming_inference. Отношение
печатается как диагностика.

    python tests/test_wkv_infer_vs_train.py
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_metal.kernel.wkv7 import wkv7_infer, wkv7_train, HEAD_SIZE  # noqa: E402

print("исполняется rwkv_metal:", sys.modules["rwkv_metal"].__file__)

# T=40 -- как в test_streaming_inference (не кратно CHUNK=16, значит
# батчевый путь паддится); 512/H=40 -- боевая форма префилла на 2.9B;
# T=7 -- короткий хвост; T=1 -- декод; B=2 ловит индексацию по батчу.
CASES = [(1, 1, 32), (1, 7, 32), (1, 40, 32), (2, 128, 32), (1, 512, 40)]
GROSS = 1e-4      # единственный порог; оба его края измерены (см. шапку)


def inputs(B, T, H, seed):
    rs = np.random.RandomState(seed)
    D = HEAD_SIZE

    def a(s):
        return (rs.randn(B, T, H, D) * s).astype(np.float32)
    # w -- готовый множитель затухания; на обученной 1.5B он не опускается
    # ниже exp(-0.606531) = 0.545 (rwkv-metal/tests/probe_w_distribution.py).
    w = (0.545 + 0.455 * rs.rand(B, T, H, D)).astype(np.float32)
    return a(.5), w, a(.5), a(.5), a(.3), a(.3)


def exact_fp64(r, w, k, v, a, b):
    """Та же рекуррентность в fp64 на CPU. Формула -- построчно из ядра:
        sa = h . a;  h = w*h + v (x) k + sa (x) b;  y = h . r
    Векторизовано по (B, H); цикл по времени -- последовательный, иначе это
    была бы другая математика, а не другая точность."""
    B, T, H, D = r.shape
    r, w, k, v, a, b = (x.astype(np.float64) for x in (r, w, k, v, a, b))
    h = np.zeros((B, H, D, D), dtype=np.float64)
    out = np.empty((B, T, H, D), dtype=np.float64)
    for t in range(T):
        sa = np.einsum("bhvk,bhk->bhv", h, a[:, t])
        h = (h * w[:, t][:, :, None, :]
             + v[:, t][:, :, :, None] * k[:, t][:, :, None, :]
             + sa[:, :, :, None] * b[:, t][:, :, None, :])
        out[:, t] = np.einsum("bhvk,bhk->bhv", h, r[:, t])
    return out


def rel(x, ref, scale):
    return float(np.abs(x - ref).max()) / scale


def main():
    ok = True
    print("%-14s %11s %11s %8s %11s" %
          ("случай", "train/fp64", "infer/fp64", "отнош.", "train/infer"))
    rows = []
    for B, T, H in CASES:
        rn, wn, kn, vn, an, bn = inputs(B, T, H, seed=1000 + T + H)
        arrs = [mx.array(x) for x in (rn, wn, kn, vn, an, bn)]

        train = np.array(wkv7_train(*arrs))
        infer, _ = wkv7_infer(*arrs, mx.zeros((B, H, HEAD_SIZE, HEAD_SIZE),
                                              dtype=mx.float32))
        infer = np.array(infer)
        ref = exact_fp64(rn, wn, kn, vn, an, bn)
        scale = float(np.abs(ref).max())

        d_tr, d_in = rel(train, ref, scale), rel(infer, ref, scale)
        pair = rel(train, infer.astype(np.float64), scale)
        rat = max(d_tr, d_in) / max(min(d_tr, d_in), 1e-30)
        good = max(d_tr, d_in) < GROSS
        ok &= good
        rows.append((T, pair))
        print("%-14s %11.3e %11.3e %7.2fx %11.3e %s"
              % ("B%d T%d H%d" % (B, T, H), d_tr, d_in, rat, pair,
                 "" if good else "<-- ВЫШЕ КРЫШКИ %.0e" % GROSS))

    # Расхождение ядер обязано расти как случайное блуждание последнего
    # бита: ~sqrt(T). Если оно растёт линейно -- это уже не округление.
    r0 = [p for t, p in rows if t == 1][0]
    print("\nрост расхождения ядер по T (норм. на T=1, ожидание sqrt(T)):")
    for t, p in rows:
        print("  T=%-4d %6.1fx   sqrt(T) = %.1f" % (t, p / max(r0, 1e-30), t ** .5))

    # НЕВЫРОЖДЕННОСТЬ (закон 8) и ЧУВСТВИТЕЛЬНОСТЬ: сравнение обязано быть
    # способно показать расхождение. Если бы ядра отдавали одинаковый мусор
    # (нули, непокрытые гридом строки), все числа выше были бы нулями.
    rn, wn, kn, vn, an, bn = inputs(1, 40, 32, seed=77)
    arrs = [mx.array(x) for x in (rn, wn, kn, vn, an, bn)]
    ref = exact_fp64(rn, wn, kn, vn, an, bn)
    scale = float(np.abs(ref).max())
    assert scale > 1e-3, "выход вырожден -- сравнивать нечего"
    infer, _ = wkv7_infer(*arrs, mx.zeros((1, 32, HEAD_SIZE, HEAD_SIZE),
                                          dtype=mx.float32))
    bad = np.array(infer).astype(np.float64) * (1.0 + 2.0 ** -8)
    ctrl = rel(bad, ref, scale)
    print("\nконтроль чувствительности: ошибка 0.39%% даёт %.3e "
          "против крышки %.0e -- запас %.0fx" % (ctrl, GROSS, ctrl / GROSS))

    print("\n[%s] wkv7_infer против wkv7_train" % ("OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
