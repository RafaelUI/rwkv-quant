# -*- coding: utf-8 -*-
"""ВХОДЫ fake-ПЛЕЧА ПРОТИВ ФАЙЛА -- НА НАСТОЯЩИХ ВЕСАХ МОДЕЛИ.

`probe_fake_vs_file*.py` сверяли РЕКОНСТРУКЦИЮ: они звали `fake_quant.q`
на тензоре ЧЕКПОИНТА. Но fake-плечо квантует не чекпоинт, а то, что лежит
в модели, а `RWKV7Ref` держит LoRA-матрицы ТРАНСПОНИРОВАННЫМИ
(`blocks.0.att.w1` -- [2048, 96] в чекпоинте и [96, 2048] в модели).
Группы идут вдоль последней оси, значит транспонирование меняет само
разбиение на группы -- и реконструкция не могла этого увидеть.

Здесь берутся РОВНО те тензоры, которые квантует `ablate_subgroups`
(через `comp.quant_points`), и сверяются с файлом, приведённым к той же
ориентации. Печатается ориентация каждой разновидности -- она и есть
предмет проверки.

    python tests/_sess/probe_arm_inputs.py <файл.rwkvq> [конфиг]
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch  # noqa: E402

import ablate_subgroups as A  # noqa: E402
import ablate_sym_composite as comp  # noqa: E402
from rwkv_quant.calibration import fake_quant  # noqa: E402
from rwkv_quant.formats import reader  # noqa: E402
from rwkv_quant.models.rwkv7_ref import RWKV7Ref  # noqa: E402

PATH = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "preset"


def main():
    ck = reader.load_raw(PATH)
    sd = torch.load(A.CKPT, map_location="cpu", mmap=True)
    cfg = comp.CONFIGS[NAME]()
    model = RWKV7Ref(A.CKPT, device="cpu", dtype=torch.float32)
    print("конфиг %s, файл %s" % (NAME, os.path.basename(os.path.realpath(PATH))),
          flush=True)

    agg = collections.defaultdict(lambda: [0.0, 0.0, 0, 0, 0, 0.0])
    rows, absent, nT = [], [], 0
    for obj, attr, group, key in comp.quant_points(model):
        w = getattr(obj, attr)
        qt = ck.tensors.get(key)
        if w is None or qt is None:
            absent.append(key)
            continue
        w = w.to(torch.float32)
        if qt.bits >= 16:
            w_file = qt.dense.to(torch.float32)
        else:
            w_file = reader.dequantize_banded(qt, torch.float32)
        orient = "прямая"
        if tuple(w.shape) != tuple(w_file.shape):
            if tuple(w.shape) == tuple(w_file.shape[::-1]):
                w_file = w_file.T.contiguous()
                orient = "ТРАНСП"
                nT += 1
            else:
                raise SystemExit("формы несводимы у %s: %s против %s"
                                 % (key, tuple(w.shape), tuple(w_file.shape)))
        # контроль: сам исходный вес модели обязан совпасть с чекпоинтом
        # в той же ориентации, иначе сверять нечего
        w_ck = sd[key].to(torch.float32)
        if orient == "ТРАНСП":
            w_ck = w_ck.T.contiguous()
        d_load = float((w - w_ck).abs().max())

        w_fake = fake_quant.q(w, group, cfg, key).to(torch.float32)
        d = w_fake - w_file
        eq2 = float((w_fake - w).pow(2).sum())
        ed2 = float(d.pow(2).sum())
        ndiff = int((d != 0).sum())
        dmax = float(d.abs().max())
        sig = (group, getattr(qt, "gw_mode", "") or "rtn", int(qt.bits), orient)
        a = agg[sig]
        a[0] += eq2
        a[1] += ed2
        a[2] += w.numel()
        a[3] += ndiff
        a[4] += 1
        a[5] = max(a[5], d_load)
        if ndiff:
            rows.append((ed2, key, sig, w.numel(), dmax, eq2))
        del w, w_file, w_fake, w_ck, d

    tot_q = sum(a[0] for a in agg.values())
    tot_d = sum(a[1] for a in agg.values())
    if tot_q == 0.0:
        raise SystemExit("ОШИБКА КОНТРОЛЯ: ошибка квантования нулевая везде")

    print("")
    print("%-30s %5s %11s %11s %10s %9s %10s"
          % ("группа/раскл/бит/ориент", "кл", "RMS кванта", "RMS плечо-файл",
             "доля RMS", "разн.эл.", "max|загр|"))
    for sig in sorted(agg, key=lambda s: -agg[s][1]):
        eq2, ed2, n, ndiff, nk, dload = agg[sig]
        rq = (eq2 / n) ** 0.5 if n else 0.0
        rd = (ed2 / n) ** 0.5 if n else 0.0
        print("%-30s %5d %11.4e %11.4e %9.1f%% %8.2f%% %10.2e"
              % ("%s/%s@%d/%s" % sig, nk, rq, rd,
                 100.0 * rd / rq if rq else float("nan"),
                 100.0 * ndiff / n if n else 0.0, dload))

    print("")
    print("ЭНЕРГИЯ: расхождение плечо-файл = %.3f%% энергии ошибки кванта"
          % (100.0 * tot_d / tot_q))
    print("транспонированных точек: %d, ключей с расхождением: %d"
          % (nT, len(rows)))
    if absent:
        print("нет в файле (%d): %s" % (len(absent), ", ".join(absent[:6])))

    rows.sort(reverse=True)
    print("")
    print("%-34s %-26s %11s %11s %9s"
          % ("ключ", "группа/раскл/бит/ориент", "RMS кванта", "RMS плечо-файл",
             "доля"))
    for ed2, key, sig, n, dmax, eq2 in rows[:20]:
        rq = (eq2 / n) ** 0.5
        rd = (ed2 / n) ** 0.5
        print("%-34s %-26s %11.4e %11.4e %8.1f%%"
              % (key[:34], "%s/%s@%d/%s" % sig, rq, rd,
                 100.0 * rd / rq if rq else float("nan")))
    if len(rows) > 20:
        print("... ещё %d ключей" % (len(rows) - 20))


if __name__ == "__main__":
    main()
