"""ГЕЙТ: калибровка меряет РОВНО ту схему, которую пишет writer.

Дефект, ради которого гейт существует (найден 10.08.2026). Критерий, по
которому `api.calibrate()` выбирает битность, считался функцией
`calibration.fake_quant.q()`. Она знала только per-row RTN, clip и SpQR
и НЕ СМОТРЕЛА на `cfg.group_scale` вообще:

    q(w, "proj", QuantConfig(proj=4))                              ==
    q(w, "proj", QuantConfig(proj=4, group_scale={"proj": 32}))     -- побитово

а пресеты REDUCTION/COMPRESSION собраны ровно на group_scale. То есть
калибровали одну схему, деплоили другую, и `calibrate()` возвращал
конфиг БЕЗ group_scale -- значит `quantize(config=calibrated)` уходил в
per-row ветку, ту самую, которую README называет сломанной (canonical
int4 -> ppl 3798).

Утверждения:
  1. q() РАЗЛИЧАЕТ конфиги с group_scale и без (мутация: если кто-то
     вернёт слепую версию, тест краснеет);
  2. для каждого режима из GW_MODES выход q() совпадает БИТ-В-БИТ с
     writer.quantize_tensor(real_gw=False).dense -- то есть с тем, что
     реально попадает в fake-артефакт;
  3. для sb6-режимов выход q() совпадает и с РЕАЛЬНОЙ упаковкой
     (real_gw=True -> деквант), то есть и с содержимым .rwkvq;
  4. кеш q() не искажает результат и не переживает замер;
  5. неприменимая форма (не 2-D при заданном group_scale) даёт ЯВНУЮ
     ошибку, а не тихий уход в per-row.

    python tests/test_calib_matches_writer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.calibration import fake_quant
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.groupwise import GW_MODES
from rwkv_quant.formats import writer
from rwkv_quant.formats.reader import _dequantize_one

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
KEY = "blocks.3.ffn.key.weight"      # cmix, форма [3072, 768] -- IN кратен 256


def _w():
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    return sd[KEY].to(torch.bfloat16).contiguous()


def _cfg(bits, mode, gs=32, act=None):
    return QuantConfig(cmix=bits, group_scale={"cmix": gs},
                       group_scale_mode={"cmix": mode}, act_stats_path=act)


def test_not_blind(w):
    """Мутационная проверка: слепая q() дала бы здесь ноль."""
    blind = QuantConfig(cmix=4)
    seeing = _cfg(4, "asym_sb6")
    a = fake_quant.q(w, "cmix", blind, KEY).float()
    b = fake_quant.q(w, "cmix", seeing, KEY).float()
    d = (a - b).abs().max().item()
    assert d > 0, ("q() НЕ РАЗЛИЧАЕТ group_scale -- это ровно тот дефект, "
                   "ради которого написан гейт")
    ea = (a - w.float()).pow(2).mean().sqrt().item()
    eb = (b - w.float()).pow(2).mean().sqrt().item()
    print(f"1. q() различает group_scale: max|Δ|={d:.4g}")
    print(f"   RMSE per-row={ea:.5g}  groupwise={eb:.5g}  "
          f"(groupwise точнее в {ea/eb:.2f}x)")
    assert eb < ea, "groupwise обязан быть точнее per-row на тех же битах"


def test_matches_writer_fake(w):
    """q() == writer.quantize_tensor(real_gw=False) для всех режимов."""
    worst = 0.0
    for mode, (needs_act, _) in GW_MODES.items():
        bitset = (5, 6) if mode.startswith("sym") else (
            (6,) if mode == "asym" else ((4,) if mode == "mxfp4" else (4, 5, 6)))
        gs = 16 if mode.startswith("sym") else 32
        for bits in bitset:
            cfg = _cfg(bits, mode, gs=gs)
            a = fake_quant.q(w, "cmix", cfg, KEY)
            qt = writer.quantize_tensor(KEY, w, cfg, real_gw=False)
            b = qt.dense
            assert a.dtype == b.dtype, f"{mode}/{bits}: {a.dtype} vs {b.dtype}"
            d = (a.float() - b.float()).abs().max().item()
            worst = max(worst, d)
            assert d == 0.0, f"{mode} bits={bits}: max|Δ|={d}"
    print(f"2. q() == writer fake-путь, все {len(GW_MODES)} режимов: "
          f"max|Δ| = {worst}")


def test_matches_real_pack(w):
    """q() == деквант РЕАЛЬНОЙ упаковки (то, что лежит в .rwkvq)."""
    worst = 0.0
    for mode in ("asym_sb6", "asym_sb6_search", "asym_sb6_aw"):
        for bits in (4, 5, 6):
            cfg = _cfg(bits, mode)
            a = fake_quant.q(w, "cmix", cfg, KEY)
            qt = writer.quantize_tensor(KEY, w, cfg, real_gw=True)
            b = _dequantize_one(qt)
            d = (a.float() - b.float()).abs().max().item()
            worst = max(worst, d)
            assert d == 0.0, f"{mode} bits={bits}: max|Δ|={d}"
    print(f"3. q() == реальная упаковка .rwkvq: max|Δ| = {worst}")


def test_cache(w):
    """Кеш обязан быть прозрачным по числам и честным по памяти.

    Политика «либо весь рабочий набор, либо ничего» -- не вкус, а вывод
    замера: при нехватке бюджета частичный кеш проигрывает отсутствию
    кеша и по времени (119 с против 90), и по свопу (+4424 МБ против +0).
    См. комментарий в fake_quant и tests/probe_ppl_memory.py.
    """
    cfg = _cfg(4, "asym_sb6_search")
    ref = fake_quant.q(w, "cmix", cfg, KEY).clone()

    fake_quant.cache_begin(budget_bytes=512 << 20)
    a1 = fake_quant.q(w, "cmix", cfg, KEY)
    a2 = fake_quant.q(w, "cmix", cfg, KEY)
    assert a1 is a2, "второй вызов обязан прийти из кеша"
    assert (a1.float() - ref.float()).abs().max().item() == 0.0
    n = len(fake_quant._CACHE)
    fake_quant.cache_end()
    assert not fake_quant._CACHE, "кеш обязан умирать вместе с замером"

    # различает конфиги, а не только тензор
    fake_quant.cache_begin(budget_bytes=512 << 20)
    b1 = fake_quant.q(w, "cmix", _cfg(4, "asym_sb6"), KEY)
    b2 = fake_quant.q(w, "cmix", _cfg(6, "asym_sb6"), KEY)
    assert (b1.float() - b2.float()).abs().max().item() > 0, \
        "кеш склеил разные битности"
    fake_quant.cache_end()

    # бюджет меньше одного тензора -> кеш СДАЁТСЯ целиком, а не вытесняет
    fake_quant.cache_begin(budget_bytes=1024)
    c1 = fake_quant.q(w, "cmix", cfg, KEY)
    st = fake_quant.cache_stats()
    assert st["gave_up"] and not fake_quant._CACHE, (
        "при нехватке бюджета кеш обязан отключиться, а не дрожать: "
        f"{st}")
    assert (c1.float() - ref.float()).abs().max().item() == 0.0, \
        "отключение кеша не смеет менять числа"
    fake_quant.cache_end()

    assert fake_quant.available_bytes() > 0, "vm_stat не читается -- кеш не включится"
    print(f"4. кеш: попадание ({n} запись), конфиги не склеены, при нехватке "
          f"бюджета отключается целиком, числа не меняются")


def test_bad_shape(w):
    cfg = QuantConfig(small=4, group_scale={"small": 32})
    w3 = w[:1, :256].reshape(1, 1, 256)
    try:
        fake_quant.q(w3, "small", cfg, "blocks.0.att.k_k")
    except ValueError as e:
        assert "2-D" in str(e)
        print("5. неприменимая форма -> явная ошибка, а не тихий per-row: OK")
        return
    raise AssertionError("ожидалась ValueError на 3-D тензоре с group_scale")


if __name__ == "__main__":
    w = _w()
    print(f"тензор: {KEY} {tuple(w.shape)} {w.dtype}\n")
    test_not_blind(w)
    test_matches_writer_fake(w)
    test_matches_real_pack(w)
    test_cache(w)
    test_bad_shape(w)
    print("\nГЕЙТ ЗЕЛЁНЫЙ")
