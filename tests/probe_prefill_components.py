"""ИЗ ЧЕГО СОСТОИТ ПРЕФИЛЛ (аблации на компилированном пути, T=512).

`profile_components` мерил ДЕКОД, и его выводы на префилл не переносятся:
там пол считается от полосы памяти, здесь -- от арифметики (веса читаются
раз на вызов, 1.5% времени). Пол по ФЛОПам при измеренном потолке
2.80 ТФЛОП/с -- 510 мс на 1.5B против измеренных 753, и вопрос ровно в
том, из чего состоят оставшиеся 32%.

Аблируются ТРИ статьи, каждая -- подменой функции ДО mx.compile (иначе
компилятор оттрассирует старую ветку) и со сбросом `_fused_built`
(фьюз кеширует КОПИИ весов и подмены не видит -- на этом уже
спотыкались):

  WKV      -- последовательная рекуррентность по чанкам, от пикового
              GEMM не ускоряется вовсе;
  LoRA     -- восемь матмулов на слой мелких форм;
  деквант  -- `_dequant_w` отдаёт нули: уходит и распаковка, и её
              трафик, а сам матмул остаётся.

Аблации НЕ аддитивны (закон: убрав компонент, меняешь и перекрытие),
инструмент ищет доминанту, а не строит бюджет.

    python tests/probe_prefill_components.py [model.rwkvq] [T] [раундов]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.backends.metal import quant_linear_sym as qls  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def main():
    model = qm.QuantRWKV7(load_raw(PATH))
    idx = mx.array(np.random.RandomState(7).randint(
        0, 65500, size=(1, T)).astype(np.int32))

    def run(fn):
        st = model.init_state(1)
        lg, st = fn(idx, st); mx.eval(lg); mx.synchronize()
        st = model.init_state(1)
        t0 = time.perf_counter()
        lg, st = fn(idx, st); mx.eval(lg)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    orig_wkv = qm._wkv_stateful
    orig_lora = qm.QuantTMix._lora
    orig_dq = qls.SymQuantLinear._dequant_w

    def ab(patch):
        """Патч ставится ДО компиляции; фьюз сбрасывается, иначе аблация
        не видна закешированным копиям."""
        for b in model.blocks:
            b.tmix._fused_built = False
        patch()
        f = mx.compile(model.forward_stateful)
        t = run(f)
        qm._wkv_stateful = orig_wkv
        qm.QuantTMix._lora = orig_lora
        qls.SymQuantLinear._dequant_w = orig_dq
        for b in model.blocks:
            b.tmix._fused_built = False
        del f
        return t

    def p_none():
        pass

    def p_wkv():
        qm._wkv_stateful = lambda r, w, k, v, a, b, st: (
            mx.zeros_like(r), st)

    def p_lora():
        qm.QuantTMix._lora = lambda self, xw, xa, xv, xg, x, xx: (
            mx.zeros(x.shape, dtype=x.dtype), mx.zeros(x.shape, dtype=x.dtype),
            None if self.v_lora_A is None and not getattr(
                self, "_dense_lora_dropped", False) else mx.zeros(
                x.shape, dtype=x.dtype),
            mx.zeros(x.shape, dtype=x.dtype))

    def p_dq():
        qls.SymQuantLinear._dequant_w = lambda self: mx.zeros(
            (self.out_features, self.in_features), dtype=mx.float16)

    sw0 = swap_mb()
    full, deltas = [], {"WKV": [], "LoRA": [], "деквант": []}
    for r in range(ROUNDS):
        t_full = ab(p_none)
        full.append(t_full)
        for name, patch in (("WKV", p_wkv), ("LoRA", p_lora),
                            ("деквант", p_dq)):
            deltas[name].append(t_full - ab(patch))
        print(f"  раунд {r}: полный {t_full:.0f} мс | "
              + "  ".join(f"{k} {v[-1]:+.0f}" for k, v in deltas.items())
              + f" | своп {swap_mb():.0f}", flush=True)
    sw1 = swap_mb()

    f = float(np.median(full))
    print(f"\nполный префилл T={T}: {f:.1f} мс")
    print(f"{'компонент':>10} | {'мс':>7} {'% шага':>8}")
    for k, v in deltas.items():
        d = float(np.median(v))
        print(f"{k:>10} | {d:6.1f} {d/f*100:7.1f}%")
    print(f"\nпол по арифметике при 2.80 ТФЛОП/с -- 510 мс на 1.5B; "
          f"остаток сверх него: {f-510:.0f} мс")
    print(f"своп: {sw0:.0f} -> {sw1:.0f} МБ "
          f"({'ОК' if sw1 - sw0 < 1 else 'ЗАМЕР НЕДЕЙСТВИТЕЛЕН'})")


if __name__ == "__main__":
    main()
