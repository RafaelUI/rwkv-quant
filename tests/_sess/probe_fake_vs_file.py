# -*- coding: utf-8 -*-
"""ГДЕ РАСХОДЯТСЯ fake-ПУТЬ И ФАЙЛ: СРАВНЕНИЕ САМИХ ВЕСОВ, А НЕ ВЫХОДА.

Лестница плеч на реальном пути (27.08) сняла ВСЕ ТРИ записанных кандидата
на остаток 0.000541 нат/ток: fp16-контейнеры, арифметика плотных матмулов
и GEMV-кернель дают вместе меньше 0.00007. Значит расходятся не вычисления,
а САМИ ЧИСЛА ВЕСОВ: fake-путь считает их `fake_quant.q`, а в файле лежит
то, что записал writer.

Проверяется прямо: для представителя КАЖДОЙ разновидности (группа,
раскладка, бит -- закон 31, а не край сортировки) берутся w_fake и w_file
и печатается RMS их расхождения рядом с RMS собственной ошибки квантования
||w_fake - w||. Отношение и есть искомое: ноль -- пути тождественны,
порядок единицы -- fake моделирует не тот формат.

    python tests/_sess/probe_fake_vs_file.py <файл.rwkvq> <чекпоинт.pth> [конфиг]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch  # noqa: E402

from rwkv_quant.formats import reader  # noqa: E402
from rwkv_quant.calibration import fake_quant  # noqa: E402
import ablate_sym_composite as comp  # noqa: E402


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def main():
    path, ckpt_path = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "preset"
    ck = reader.load_raw(path)
    sd = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", mmap=True)
    cfg = comp.CONFIGS[name]()
    print("конфиг %s, файл %s" % (name, os.path.basename(os.path.realpath(path))))

    seen, picks = set(), []
    for key, qt in ck.tensors.items():
        if qt.bits >= 16 or key not in sd:
            continue
        sig = (getattr(qt, "group", "?"), getattr(qt, "gw_mode", "") or "rtn",
               qt.bits)
        if sig in seen:
            continue
        seen.add(sig)
        picks.append((sig, key, qt))

    print("%-30s %-20s %11s %11s %8s"
          % ("ключ", "группа/раскл/бит", "RMS кванта", "RMS fake-file", "доля"))
    tot_q = tot_d = 0.0
    for sig, key, qt in picks:
        w = sd[key].to(torch.float32)
        w_file = reader.dequantize_banded(qt, torch.float32)
        w_fake = fake_quant.q(sd[key], sig[0], cfg, key).to(torch.float32)
        e_q, e_d = rms(w_fake - w), rms(w_fake - w_file)
        tot_q += e_q; tot_d += e_d
        print("%-30s %-20s %11.4e %11.4e %7.1f%%"
              % (key[:30], "%s/%s@%d" % sig, e_q, e_d,
                 100.0 * e_d / e_q if e_q else float("nan")))
        del w, w_file, w_fake
    if tot_q == 0.0:
        raise SystemExit("ОШИБКА КОНТРОЛЯ: ошибка квантования нулевая во всех "
                         "плечах -- fake_quant.q ничего не сделал")


if __name__ == "__main__":
    main()
