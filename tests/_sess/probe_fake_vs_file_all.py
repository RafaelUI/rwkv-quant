# -*- coding: utf-8 -*-
"""ПОСЛОЙНАЯ СВЕРКА ВЕСОВ fake-ПУТИ ПРОТИВ ФАЙЛА: ВСЕ КЛЮЧИ, А НЕ ПРЕДСТАВИТЕЛИ.

`probe_fake_vs_file.py` брал ОДНОГО представителя на разновидность
(группа/раскладка/бит), то есть фактически слой 0: закон 31 выполнен по
разновидностям и не выполнен по слоям. Здесь сверяются ВСЕ ключи
контейнера, и в том числе ПЛОТНЫЕ (bits>=16), которых прежний зонд не
смотрел вовсе -- а их 460 из 798, и в fake-пути они идут прямо из
чекпоинта.

Считается энергия расхождения (сумма квадратов), а не только RMS: доли
энергии складываются, RMS -- нет, а вопрос стоит именно «чья это доля
остатка 0.001216».

    python tests/_sess/probe_fake_vs_file_all.py <файл.rwkvq> <чекпоинт.pth> [конфиг]
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch  # noqa: E402

from rwkv_quant.formats import reader  # noqa: E402
from rwkv_quant.calibration import fake_quant  # noqa: E402
import ablate_sym_composite as comp  # noqa: E402


def main():
    path, ckpt_path = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "preset"
    ck = reader.load_raw(path)
    sd = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", mmap=True)
    cfg = comp.CONFIGS[name]()
    print("конфиг %s, файл %s, тензоров в контейнере %d"
          % (name, os.path.basename(os.path.realpath(path)), len(ck.tensors)))

    # eq2 -- энергия ошибки квантования, ed2 -- энергия расхождения fake/file
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0, 0, 0, 0.0])
    rows = []
    miss = []
    for key, qt in ck.tensors.items():
        if key not in sd:
            miss.append(key)
            continue
        grp = getattr(qt, "group", "?")
        mode = getattr(qt, "gw_mode", "") or "rtn"
        w = sd[key].to(torch.float32)
        if qt.bits >= 16:
            w_file = qt.dense.to(torch.float32)
            w_fake = w
        else:
            w_file = reader.dequantize_banded(qt, torch.float32)
            w_fake = fake_quant.q(sd[key], grp, cfg, key).to(torch.float32)
        d = w_fake - w_file
        eq2 = float((w_fake - w).pow(2).sum())
        ed2 = float(d.pow(2).sum())
        ndiff = int((d != 0).sum())
        dmax = float(d.abs().max())
        sig = (grp, mode, int(qt.bits))
        a = agg[sig]
        a[0] += eq2
        a[1] += ed2
        a[2] += w.numel()
        a[3] += ndiff
        a[4] += 1
        a[5] = max(a[5], dmax)
        if ndiff:
            rows.append((ed2, key, sig, ndiff, w.numel(), dmax, eq2))
        del w, w_file, w_fake, d

    tot_q = sum(a[0] for a in agg.values())
    tot_d = sum(a[1] for a in agg.values())
    if tot_q == 0.0:
        raise SystemExit("ОШИБКА КОНТРОЛЯ: ошибка квантования нулевая везде -- "
                         "fake_quant.q ничего не сделал")

    print("")
    print("%-22s %5s %11s %11s %11s %9s %11s"
          % ("группа/раскл/бит", "кл", "RMS кванта", "RMS fake-file",
             "доля RMS", "разн.эл.", "max|d|"))
    for sig in sorted(agg, key=lambda s: -agg[s][1]):
        eq2, ed2, n, ndiff, nk, dmax = agg[sig]
        rq = (eq2 / n) ** 0.5 if n else 0.0
        rd = (ed2 / n) ** 0.5 if n else 0.0
        print("%-22s %5d %11.4e %11.4e %10.1f%% %8.2f%% %11.4e"
              % ("%s/%s@%d" % sig, nk, rq, rd,
                 100.0 * rd / rq if rq else float("nan"),
                 100.0 * ndiff / n if n else 0.0, dmax))

    print("")
    print("ЭНЕРГИЯ: расхождение fake-file = %.4f%% энергии ошибки квантования"
          % (100.0 * tot_d / tot_q))
    print("ключей с ненулевым расхождением: %d из %d" % (len(rows), len(ck.tensors)))
    if miss:
        print("нет в чекпоинте (%d): %s" % (len(miss), ", ".join(sorted(miss)[:6])))

    rows.sort(reverse=True)
    print("")
    print("%-34s %-20s %11s %11s %9s"
          % ("ключ", "группа/раскл/бит", "RMS кванта", "RMS fake-file", "доля"))
    for ed2, key, sig, ndiff, n, dmax, eq2 in rows[:24]:
        rq = (eq2 / n) ** 0.5
        rd = (ed2 / n) ** 0.5
        print("%-34s %-20s %11.4e %11.4e %8.1f%%"
              % (key[:34], "%s/%s@%d" % sig, rq, rd,
                 100.0 * rd / rq if rq else float("nan")))
    if len(rows) > 24:
        print("... ещё %d ключей" % (len(rows) - 24))

    # разброс доли ПО СЛОЯМ внутри каждой разновидности: 13.3% на слое 0 --
    # это про слой 0, пока не показан размах.
    per = collections.defaultdict(list)
    for ed2, key, sig, ndiff, n, dmax, eq2 in rows:
        if eq2:
            per[sig].append(100.0 * ((ed2 / n) ** 0.5) / ((eq2 / n) ** 0.5))
    if per:
        print("")
        print("РАЗМАХ ДОЛИ ПО СЛОЯМ (только ключи с расхождением):")
        for sig in sorted(per, key=lambda s: -max(per[s])):
            v = sorted(per[sig])
            print("  %-22s n=%3d  мин %.1f%%  медиана %.1f%%  макс %.1f%%"
                  % ("%s/%s@%d" % sig, len(v), v[0], v[len(v) // 2], v[-1]))


if __name__ == "__main__":
    main()
