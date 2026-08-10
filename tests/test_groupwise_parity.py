"""ГЕЙТ: перенос блочных fake-quant схем из formats/writer.py в
calibration/groupwise.py не изменил ни одного бита.

Три независимых утверждения:

  1. ТОЖДЕСТВО ИМЁН. writer._groupwise_fake_dequant IS
     groupwise.groupwise_fake_dequant. Это сильнее численной сверки:
     разойтись физически нечему, потому что объект один. Если кто-то
     вернёт в writer собственную копию, тест покраснеет сразу, а не
     после расхождения на четвёртом знаке.

  2. ЭТАЛОН. Хеши выходов на реальных тензорах чекпоинта и на синтетике
     совпадают с замороженными ДО правки (tests/make_gw_golden.py).
     Реальные тензоры обязательны: закон 17 — синтетика может описывать
     несуществующую раскладку.

  3. СВЯЗЬ С УПАКОВКОЙ. Реальный упаковщик _make_qt_gw_sb6 берёт parts у
     этой же функции; деквант упакованного результата обязан совпасть с
     fake-выходом. Иначе «измеряем одно, пишем другое» вернётся с другой
     стороны.

Сверка ПОТОКОВАЯ и по хешам — см. докстринг make_gw_golden.py о том,
почему держать эталон и пересчёт одновременно однажды стоило свопа и
паники системы.

    python tests/test_groupwise_parity.py
    python tests/test_groupwise_parity.py --explain <имя случая>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from rwkv_quant.calibration import groupwise as gw
from rwkv_quant.formats import writer

import make_gw_golden as golden

GOLDEN = golden.GOLDEN
FNS = (gw.groupwise_fake_dequant, gw.groupwise_sym_fake_dequant,
       gw.mxfp4_fake_dequant)


def test_identity():
    assert writer._groupwise_fake_dequant is gw.groupwise_fake_dequant
    assert writer._groupwise_sym_fake_dequant is gw.groupwise_sym_fake_dequant
    assert writer._mxfp4_fake_dequant is gw.mxfp4_fake_dequant
    assert writer._get_ex2 is gw.get_ex2
    assert writer._load_act_stats is gw.load_act_stats
    print("1. тождество имён writer <-> groupwise: OK")


def test_golden():
    if not os.path.exists(GOLDEN):
        print(f"2. эталон {GOLDEN} отсутствует — создать: "
              f"python tests/make_gw_golden.py (на НЕТРОНУТОМ дереве)")
        return False
    ref = json.load(open(GOLDEN))["hashes"]
    sd = golden.load_sd()
    seen, bad = 0, []
    for name, t in golden.iter_results(FNS, sd):
        h = golden.digest(t)
        del t                       # ключевое: выход не переживает итерацию
        seen += 1
        if name not in ref:
            bad.append((name, "нет в эталоне"))
        elif ref[name] != h:
            bad.append((name, "хеш разошёлся"))
    missing = sorted(set(ref) - set())
    assert seen == len(ref), f"случаев сейчас {seen}, в эталоне {len(ref)}"
    assert not bad, ("расхождение с эталоном:\n  " +
                     "\n  ".join(f"{n}: {why}" for n, why in bad[:10]) +
                     f"\n(всего {len(bad)}; разбор: --explain <имя>)")
    print(f"2. эталон: {seen} случаев, все хеши совпали (бит-в-бит)")
    return True


def test_pack_roundtrip():
    """parts из groupwise -> реальная упаковка -> деквант == fake-выход.

    Сверка в bf16, и это не послабление. `_dequantize_gw_sb6` завершает
    каст в bf16, а fake-ветка `quantize_tensor` хранит ровно
    `deq.to(torch.bfloat16)` -- то есть в обоих путях наружу выходит bf16,
    и сравнивать fp32-промежуток с bf16-результатом значило бы мерить
    ширину мантиссы, а не раскладку. Первый прогон этого гейта именно так
    и упал (max|Δ| = 2.44e-4 = один бит bf16 на значениях ~0.1), и это
    был дефект теста, а не кода.
    """
    from rwkv_quant.formats.reader import _dequantize_one
    g = torch.Generator().manual_seed(11)
    sd = golden.load_sd()
    cases = [("real", sd["blocks.3.ffn.key.weight"].float()),
             ("synth", torch.randn(64, 512, generator=g))]
    worst = 0.0
    for tag, w in cases:
        ex2 = torch.rand(w.shape[-1], generator=g) + 0.05
        for bits in (4, 5, 6):
            for mode, search, e in (("asym_sb6", False, None),
                                    ("asym_sb6_search", True, None),
                                    ("asym_sb6_aw", True, ex2)):
                qt = writer._make_qt_gw_sb6("k", "cmix", bits, w, 32, e,
                                            search=search)
                deq = _dequantize_one(qt)                       # bf16
                fake = gw.groupwise_fake_dequant(
                    w, bits, 32, sb=8, sb_bits=(-6 if search else 6),
                    ex2=e).to(torch.bfloat16)                   # как в writer
                d = (deq.float() - fake.float()).abs().max().item()
                worst = max(worst, d)
                assert d == 0.0, f"{tag} {mode} bits={bits}: max|Δ|={d}"
                del qt, deq, fake
    print(f"3. fake == реальная упаковка -> деквант: max|Δ| = {worst}")


def explain(target):
    """Пересчитать ОДИН случай и показать его — то, что теряется при
    сверке по хешам, возвращается здесь и ровно для одного тензора."""
    sd = golden.load_sd()
    for name, t in golden.iter_results(FNS, sd):
        if name == target:
            print(f"{name}\n  shape={tuple(t.shape)} dtype={t.dtype}")
            print(f"  sha256={golden.digest(t)}")
            f = t.flatten().float()
            print(f"  min={f.min():.6g} max={f.max():.6g} "
                  f"mean={f.mean():.6g} первые={f[:6].tolist()}")
            return
        del t
    print(f"случай {target!r} не найден")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--explain":
        explain(sys.argv[2])
        sys.exit(0)
    print(f"своп до:    {golden.swapusage()}")
    test_identity()
    ok = test_golden()
    test_pack_roundtrip()
    print(f"своп после: {golden.swapusage()}")
    print("\nГЕЙТ ЗЕЛЁНЫЙ" + ("" if ok else " (эталон пропущен)"))
