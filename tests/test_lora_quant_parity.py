"""ГЕЙТ КВАНТОВАНИЯ LoRA-ВЕТОК (LORA_Q = "sep" / "glue").

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ И ЧЕГО ЗДЕСЬ НЕ ПРОВЕРЯЕТСЯ. Это НЕ репак: в файле
LoRA лежит в asym gw64, но группы там вдоль ВЫХОДНОЙ оси матмула (writer
квантует по сырым ключам, до транспозиции), а mx.quantized_matmul требует
их вдоль ВХОДНОЙ. Значит значения МЕНЯЮТСЯ, требовать равенства нельзя, и
качество этим гейтом не закрывается -- его меряет tests/eval_lora_quant.py
на мультиязычном корпусе. Здесь проверяется, что раскладка собрана верно
(порядок веток, срезы склейки, кратности групп) и что расхождение лежит на
уровне арифметики шести бит, а не на уровне перепутанных строк.

ЧЕТЫРЕ УТВЕРЖДЕНИЯ:

  1. fp16-путь бит-в-бит равен себе при LORA_Q=None (защита от того, что
     общая реализация _lora что-то поменяла в непричастной ветке);
  2. квантованный путь ДЕЙСТВИТЕЛЬНО включился -- буферы построены И
     выход отличается от fp16. Без этой проверки гейт сравнивал бы fp16
     с fp16 и был бы зелёным всегда;
  3. relmax логитов против fp16 и совпадение greedy-траектории;
  4. "glue" против "sep": склейка обязана давать ТОТ ЖЕ порядок веток
     (перепутанный срез даёт правдоподобные числа -- ловушка из закона 8),
     поэтому сверяется отдельно и с более жёстким порогом.

    python tests/test_lora_quant_parity.py [model.rwkvq]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
NGEN = 48
PROMPT = [187, 4211, 33, 900, 12, 65535, 7, 4096]
# Порог ловит НЕ сетку кодов, а перепутанный порядок строк/срезов: там
# ошибка была бы порядка самой величины (O(1)), тогда как честное
# шестибитное квантование LoRA даёт 2e-2 на логитах префилла и 4e-3 на
# декоде -- это воспроизводит записанную относительную ошибку весов
# asym@6 (1.94e-2), то есть ветки отрабатывают ровно на свою битность.
TOL_LOGITS = {8: 1.2e-2, 6: 4.0e-2}


def run(model, fuse):
    qm.FUSE = fuse
    st = model.init_state(1)
    idx = mx.array(np.array([PROMPT], dtype=np.int32))
    logits, st = model.forward_stateful(idx, st)
    pre = np.array(logits.astype(mx.float32))
    tok = mx.argmax(logits[:, -1], axis=-1)
    seq = []
    for _ in range(NGEN):
        logits, st = model.forward_stateful(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(tok)
        seq.append(int(np.array(tok)[0]))
    return pre, np.array(logits.astype(mx.float32)), np.array(seq)


def relmax(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))


def main():
    model = qm.QuantRWKV7(load_raw(PATH))
    ok = True

    qm.LORA_Q = None
    base = {f: run(model, f) for f in (False, True)}
    base2 = {f: run(model, f) for f in (False, True)}
    for f in (False, True):
        d = max(relmax(base[f][i], base2[f][i]) for i in (0, 1))
        print(f"[1] fp16 сам с собой, FUSE={f}: max rel {d:.1e} "
              f"{'OK' if d == 0 else 'НЕ ДЕТЕРМИНИРОВАН'}")
        ok &= d == 0

    for bits in (8, 6):
        for mode in ("sep", "glue"):
            qm.LORA_Q, qm.LORA_QBITS = mode, bits
            n = qm.reset_lora_q(model)          # закон: буферы кешируются
            got = {f: run(model, f) for f in (False, True)}
            built = sum(b.tmix._lq_A is not None for b in model.blocks)
            print(f"\n--- LORA_Q={mode} bits={bits} "
                  f"(сброшено слоёв {n}, построено {built}/{len(model.blocks)})")
            if built != len(model.blocks):
                print("    [2] КРАСНЫЙ: буферы не построены, "
                      "путь молча остался fp16")
                ok = False
                continue
            for f in (False, True):
                rp = relmax(got[f][0], base[f][0])
                rd = relmax(got[f][1], base[f][1])
                same = int((got[f][2] == base[f][2]).sum())
                acted = rp > 0
                print(f"    FUSE={f}: [2] включилось {acted} | "
                      f"[3] relmax префилл {rp:.2e} декод {rd:.2e} | "
                      f"greedy {same}/{NGEN}")
                tol = TOL_LOGITS[bits]
                ok &= acted and rp < tol and rd < tol
                if same != NGEN:
                    print(f"        greedy разошёлся на {NGEN - same} "
                          f"позициях -- само по себе не приговор (48 шагов "
                          f"без учителя), но записать")
            # [4] склейка против порознь: тот же порядок веток
            if mode == "glue":
                qm.LORA_Q = "sep"
                qm.reset_lora_q(model)
                sep = run(model, False)
                qm.LORA_Q = "glue"
                qm.reset_lora_q(model)
                gl = run(model, False)
                r = relmax(gl[0], sep[0])
                # склейка и порознь -- две сетки одного порядка, поэтому
                # между собой они обязаны сходиться ЛУЧШЕ, чем каждая с
                # fp16: расхождение уровня fp16-разности означало бы, что
                # срезы склейки разъехались по веткам
                tol = TOL_LOGITS[bits] / 2
                print(f"    [4] glue против sep: relmax {r:.2e} "
                      f"{'OK' if r < tol else 'ПОРЯДОК ВЕТОК?'}")
                ok &= r < tol

    qm.LORA_Q = None
    print(f"\nГЕЙТ: {'ЗЕЛЁНЫЙ' if ok else 'КРАСНЫЙ'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
