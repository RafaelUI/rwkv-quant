"""A/B: Metal-GEMV раскладки sym (Q6_K) против sb6 на ОДНИХ И ТЕХ ЖЕ
реальных формах и одних и тех же весах.

ЗАЧЕМ. Упаковщик и кернель sym написаны и численно сверены, композит
показал, что бюджет REDUCTION падает вчетверо-впятеро, -- но скорость не
мерена ВООБЩЕ. Пресет с sym меняет кернель у cmix, proj и head, то есть
у 88% файла. Если sym медленнее, правка пресета -- это обмен качества на
токены в секунду по НЕИЗВЕСТНОМУ курсу, а декод в этом проекте закрыт
как «достаточно» на конкретном числе (54.86 ток/с у REDUCTION), а не «как
получится».

Априори у sym две противонаправленные силы, и какая перевесит -- вопрос
замера, а не рассуждения:
  ПРОТИВ: блок вдвое уже (16 против 32), значит масштабов на строку
    вдвое больше -- лишний трафик и лишние лоады;
  ЗА: min нет вовсе, то есть на блок не читается ни qm, ни dm, и из
    внутреннего цикла уходит одно умножение с накоплением; при восьми
    битах ещё и распаковки нет -- код это байт.

МЕТОДОЛОГИЯ (закон 1). Только чередование A/B в ОДНОМ процессе:
безвентиляторный M4 даёт дрейф до 1.8x на «том же» замере между
процессами. Синк амортизируется цепочкой из восьми вызовов и вычитается
отдельным замером. Медиана по повторам, повторы чередуются по кейсам.

СВОП (закон 11). Малейший рост свопа делает замер скорости
недействительным. Фиксируется до и после; при росте отчёт помечается
НЕДЕЙСТВИТЕЛЬНЫМ, а не печатается как ни в чём не бывало.

    python tests/bench_sym_vs_sb6_ab.py [ckpt]          # A/B
    python tests/bench_sym_vs_sb6_ab.py [ckpt] --sweep  # (NSG, RS)
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal.quant_linear_gw import GwQuantLinear  # noqa: E402
from rwkv_quant.backends.metal.quant_linear_sym import SymQuantLinear  # noqa: E402
from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SWEEP = "--sweep" in sys.argv
CKPT = ARGS[0] if ARGS else os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
ACT = os.environ.get("RWKVQ_ACT_STATS", "/tmp/act_stats_1p5b_ml.pt")

CASES = [
    ("cmix key",   "blocks.0.ffn.key.weight",        "cmix"),
    ("cmix value", "blocks.0.ffn.value.weight",      "cmix"),
    ("proj",       "blocks.0.att.receptance.weight", "proj"),
    ("head",       "head.weight",                    "head"),
]


def swap_mb():
    """Своп в МБ. LC_ALL=C ОБЯЗАТЕЛЕН: в локали с запятичным разделителем
    sysctl печатает "532,75M", и float() на этом падает. У автора скрипта
    оболочка была в C-локали, поэтому баг не воспроизводился -- классика
    «работает на моей машине». Запятая на всякий случай тоже разбирается."""
    env = dict(os.environ, LC_ALL="C", LANG="C")
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                       capture_output=True, text=True).stdout
    u = o.split("used =")[1].split()[0]          # "532.75M" | "1.50G"
    unit, num = u[-1], u[:-1]
    if "," in num:                               # запятая -- десятичная
        num = num.replace(".", "").replace(",", ".")
    return float(num) * (1024 if unit == "G" else 1)


def buf_mb(lin):
    tot = 0
    for n in ("qblk", "qs", "d", "qsqm", "ddm"):
        a = getattr(lin, n, None)
        if a is not None:
            tot += a.size * a.dtype.size
    return tot / 1e6


def bench(fn, reps=9, warm=3):
    for _ in range(warm):
        mx.eval(*fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def build(w, key, group, mode, bits):
    """Ключ передаётся НАСТОЯЩИЙ, а не "bench": по нему writer определяет
    группу (_match_group) и адресует статистику активаций. С выдуманным
    ключом группа не находится вовсе, и quantize_tensor молча возвращает
    dense bf16 -- то есть замер сравнивал бы не то, что заказан."""
    sym = mode.startswith("sym")
    cfg = QuantConfig(**{group: bits},
                      group_scale={group: 16 if sym else 32},
                      group_scale_mode={group: mode},
                      act_stats_path=ACT if os.path.exists(ACT) else None)
    qt = quantize_tensor(key, w, cfg, real_gw=True)
    assert qt.bits < 16 and qt.gw_mode, (
        f"{key}: quantize_tensor вернул bits={qt.bits} "
        f"gw_mode={qt.gw_mode!r} -- группа не совпала?")
    return SymQuantLinear(qt) if sym else GwQuantLinear(qt)


def main():
    sw0 = swap_mb()
    print(f"своп до: {sw0:.1f} МБ", flush=True)
    sd = torch.load(CKPT, map_location="cpu", mmap=True)

    rows = []
    for label, key, group in CASES:
        if key not in sd:
            continue
        w = sd[key].contiguous()
        OUT, IN = w.shape
        lins = {"sb6@6": build(w, key, group, "asym_sb6_aw", 6),
                "sym@6": build(w, key, group, "sym_aw", 6),
                "sym@8": build(w, key, group, "sym_aw", 8)}
        x1 = mx.array(np.random.randn(1, IN).astype(np.float32))
        x4 = mx.array(np.random.randn(4, IN).astype(np.float32))
        mx.eval(x1, x4)

        if SWEEP:
            # Свипается И шесть, И восемь бит: пути в кернеле разные (при
            # восьми битах нет ни битплоскостей, ни xbsum), и переносить
            # оптимум с одного на другой -- та же ошибка, что переносить
            # вывод между масштабами.
            combos = [(n, r) for n in (2, 4, 8) for r in (2, 4, 8)
                      if OUT % (n * r) == 0]
            for name in ("sym@6", "sym@8"):
                print(f"\n=== свип {name}: {label} [{OUT}, {IN}] ===",
                      flush=True)
                lin = lins[name]
                # КОНФИГИ ЧЕРЕДУЮТСЯ, а не гоняются подряд. Первая версия
                # мерила их по очереди, и два прогона подряд назвали
                # лучшими РАЗНЫЕ конфигурации (cmix key: (8,2) 1.475 против
                # (8,4) 1.433) -- это дрейф безвентиляторной машины, а не
                # свойство кернеля. Закон 1 писан ровно про это, и свип --
                # такой же замер скорости, как любой другой.
                acc = {c: [] for c in combos}
                for _ in range(5):
                    for c in combos:
                        lin.cfg_override = c
                        acc[c].append(bench(lambda: [lin(x1) for _ in range(8)]))
                lin.cfg_override = None
                med = {c: float(np.median(v)) for c, v in acc.items()}
                best = min(med, key=med.get)
                for c in combos:
                    spread = (max(acc[c]) - min(acc[c])) / med[c]
                    print(f"  NSG={c[0]} RS={c[1]}: {med[c]:7.3f} мс  "
                          f"{buf_mb(lin)*8/med[c]:6.1f} ГБ/с  "
                          f"разброс {100*spread:4.1f}%"
                          + ("  <-- лучший" if c == best else ""), flush=True)
                # «лучший» имеет смысл только если он оторвался от второго
                # больше, чем шумит сам
                rest = sorted(v for c, v in med.items() if c != best)
                gap = 100 * (rest[0] - med[best]) / med[best] if rest else 0
                print(f"  -> ({best[0]}, {best[1]}) {med[best]:.3f} мс, "
                      f"отрыв от второго {gap:.1f}%"
                      + ("" if gap > 3 else "  -- В ПРЕДЕЛАХ ШУМА, "
                         "выбирать можно любой из верхних"))
            continue

        # синк меряется РЯДОМ с кейсами, а не один раз в начале: он тоже
        # дрейфует, а вычитается из каждого числа
        def sync_only():
            return [x1 + 1.0]

        cases = {}
        for n, l in lins.items():
            cases[f"{n} N=1"] = (lambda l=l: [l(x1) for _ in range(8)],
                                 buf_mb(l) * 8)
            cases[f"{n} N=4"] = (lambda l=l: [l(x4) for _ in range(8)],
                                 buf_mb(l) * 8)
        acc = {k: [] for k in cases}
        acc["sync"] = []
        for _ in range(5):                       # ЧЕРЕДОВАНИЕ, закон 1
            for k, (fn, _mb) in cases.items():
                acc[k].append(bench(fn))
            acc["sync"].append(bench(sync_only, reps=30))
        sync = float(np.median(acc["sync"]))
        print(f"\n=== {label} [{OUT}, {IN}] === (хост-синк {sync:.3f} мс)",
              flush=True)
        print(f"{'вариант':12s} {'мс':>8s} {'МБ':>8s} {'ГБ/с':>7s} {'к sb6':>8s}")
        ref = {}
        for k, (fn, mb) in cases.items():
            t = float(np.median(acc[k])) - sync
            nn = k.split()[-1]
            if k.startswith("sb6"):
                ref[nn] = t
            rel = f"{ref[nn]/t:7.3f}x" if nn in ref else "       -"
            print(f"{k:12s} {t:8.3f} {mb:8.1f} {mb/t:7.1f} {rel:>8s}")
            rows.append((label, k, t, mb, ref.get(nn)))
        del lins, x1, x4

    sw1 = swap_mb()
    print(f"\nсвоп после: {sw1:.1f} МБ (было {sw0:.1f})")
    if sw1 > sw0 + 0.5:
        print("*** ЗАМЕР НЕДЕЙСТВИТЕЛЕН: своп вырос (закон 11) ***")
        return 1
    if not SWEEP:
        print("\nитог: >1x у столбца «к sb6» = sym БЫСТРЕЕ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
