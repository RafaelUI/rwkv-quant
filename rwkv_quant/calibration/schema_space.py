"""Пространство поиска калибровки: какие схемы вообще ПРИМЕНИМЫ к
конкретной группе конкретного чекпоинта, и сколько бит на вес каждая
реально стоит.

Зачем отдельный файл. Прежний `api.calibrate()` перебирал только
битность из (8,6,4,2) и молча предполагал per-row RTN. Схема при этом
выбиралась не поиском, а тем, что `fake_quant.q()` умела -- то есть
никак. Здесь применимость выводится ИЗ ФОРМ ТЕНЗОРОВ чекпоинта, а не из
таблицы имён:

  sb6   требует bits in (4,5,6) и IN % 256 == 0 (блок 32, суперблок 8:
        `_make_qt_gw_sb6` ассертит IN % gs == 0 и (IN//gs) % 8 == 0).
        Полноранговые proj/cmix/emb/head проходят, LoRA -- нет:
        `blocks.N.att.w1` имеет IN=64 при gs=32, то есть два блока на
        суперблок из восьми.
  asym  блок gs без суперблока, fp32 scale/min на блок; bits 5..8
        (`_make_qt_gw_asym`). Это то, чем пресеты берут LoRA-ветки.
  rtn   per-row, всегда применим -- последнее прибежище.

ЦЕНА В БИТАХ НА ВЕС -- НЕ номинальная битность, а измеренная: сумма
байт всех буферов, которые writer реально кладёт на диск, делённая на
число весов. Числа ниже сняты на blocks.3.ffn.key.weight [3072, 768]
(2.36M весов) и воспроизводятся `python tests/probe_schema_cost.py`:

  sb6@4   4.500      sb6@5   5.500      sb6@6   6.500
  asym@5  9.000      asym@6  9.000      asym@8  9.000
  rtn@4   4.021      rtn@6   8.021      rtn@8   8.021

Из этой таблицы следуют два факта, которых не было ни в README, ни в
NEXT_SESSION, и оба меняют решения:

  1. asym-ветка НЕ ЗАВИСИТ ОТ БИТНОСТИ ПО РАЗМЕРУ. Коды лежат в uint8
     (`_make_qt_gw_asym` отдаёт `codes=parts["q"]` как есть), scale и min
     -- fp32 на блок из 64. Значит 8 бит кода + 64/64 = 9.000, и `bits`
     здесь чистый параметр КАЧЕСТВА. В пресетах стоит w_lora=a_lora=
     v_lora=6, и это ровно тот же файл, что при 8, только хуже. Поэтому
     в поиске asym остался ОДИН кандидат -- на восьми битах: остальные
     не могут выиграть ни байта, а стоили бы по прогону ppl каждый.
  2. rtn БЕЗ СУБ-БАЙТОВОЙ УПАКОВКИ ВЫШЕ ЧЕТЫРЁХ БИТ. `_make_qt` пакует
     нибблы только при bits <= 4, дальше кладёт int8. То есть rtn@6 и
     rtn@8 -- один и тот же размер, и «понизить битность до шести» там
     не экономит ничего. Это тезис README про canonical int6,
     подтверждённый теперь и для внутреннего пути.
"""

SB6_MODES = ("asym_sb6", "asym_sb6_search", "asym_sb6_aw")
SB6_BITS = (4, 5, 6)
# см. пункт 1 в докстринге: при 5..8 размер одинаков, значит смысл имеет
# только максимум
ASYM_BITS = (8,)
# см. пункт 2: 5..7 неотличимы по размеру от 8
RTN_BITS = (8, 4)

SB6_COST = {4: 4.5, 5: 5.5, 6: 6.5}
ASYM_COST = 9.0


def _rtn_cost(bits, in_features):
    """Нибблы только при bits <= 4 (`_make_qt`), иначе int8 -- плюс fp16
    scale на строку."""
    return (4.0 if bits <= 4 else 8.0) + 16.0 / max(in_features, 1)

SB6_GS, SB6_SB = 32, 8
ASYM_GS = 64


class Candidate:
    """Одна точка пространства поиска: (семейство, битность, режим)."""

    __slots__ = ("family", "bits", "mode", "gs", "eff_bits")

    def __init__(self, family, bits, mode, gs, eff_bits):
        self.family, self.bits, self.mode = family, bits, mode
        self.gs, self.eff_bits = gs, eff_bits

    def apply_to(self, cfg_kwargs, group):
        """Дописать себя в набор аргументов QuantConfig."""
        cfg_kwargs["bits"][group] = self.bits
        if self.family == "sb6":
            cfg_kwargs["group_scale"][group] = self.gs
            cfg_kwargs["group_scale_mode"][group] = self.mode
        elif self.family == "asym":
            cfg_kwargs["group_scale"][group] = self.gs
            cfg_kwargs["group_scale_mode"][group] = "asym"
        else:                      # rtn -- отсутствие group_scale и есть режим
            cfg_kwargs["group_scale"].pop(group, None)
            cfg_kwargs["group_scale_mode"].pop(group, None)

    def __repr__(self):
        return (f"{self.family}@{self.bits}"
                + (f"/{self.mode}" if self.family == "sb6" else "")
                + f" ({self.eff_bits:.3f} бит/вес)")


def sb6_applicable(in_features: int) -> bool:
    return in_features % (SB6_GS * SB6_SB) == 0


def asym_applicable(in_features: int) -> bool:
    return in_features >= ASYM_GS


def candidates_for(in_features: int, have_act_stats: bool = False,
                   all_2d: bool = True):
    """Все применимые точки, отсортированные ПО ВОЗРАСТАНИЮ цены.

    Поиск идёт от дешёвого к дорогому и останавливается на первом, что
    влезло в бюджет качества -- поэтому порядок здесь и есть политика
    «минимальный размер при заданной деградации».

    have_act_stats=False выкидывает AW-режимы, и это не экономия ради
    экономии: без статистики `asym_sb6_aw` ВЫРОЖДАЕТСЯ в
    `asym_sb6_search` (см. groupwise.get_ex2), то есть измерялся бы
    дважды один и тот же конфиг. Полчаса прогонов на 1.5B за нулевую
    информацию.
    """
    if not all_2d:
        # В группе есть тензор не-2-D (`small` -- это r_k [H, N] ВМЕСТЕ с
        # k_k/k_a формы (1,1,C)). Блочные схемы там неопределены, а
        # per-row ниже пяти бит ПАДАЕТ: `_make_qt` зовёт pack_int4, а тот
        # распаковывает ровно две размерности ("too many values to
        # unpack"). Прежний calibrate перебирал (8,6,4,2) и на чекпоинте,
        # где small переживает INT2 (по README -- 61M), выбрал бы двойку,
        # после чего quantize() падал бы при упаковке. Здесь такой
        # кандидат просто не существует.
        return [Candidate("rtn", 8, "rtn", 0, _rtn_cost(8, in_features))]

    out = []
    if sb6_applicable(in_features):
        modes = SB6_MODES if have_act_stats else SB6_MODES[:2]
        for bits in SB6_BITS:
            for mode in modes:
                out.append(Candidate("sb6", bits, mode, SB6_GS, SB6_COST[bits]))
    if asym_applicable(in_features):
        for bits in ASYM_BITS:
            out.append(Candidate("asym", bits, "asym", ASYM_GS, ASYM_COST))
    for bits in RTN_BITS:
        out.append(Candidate("rtn", bits, "rtn", 0, _rtn_cost(bits, in_features)))
    # при равной цене предпочитаем блочную схему и большую битность:
    # ties -- это ровно случай asym@5..8 и rtn@6..8, где дешевле не станет,
    # а качество отличается
    out.sort(key=lambda c: (c.eff_bits, c.family == "rtn", -c.bits))
    return out


def group_shapes(state_dict, group, patterns):
    """(мин. число входных каналов, все ли тензоры группы 2-D).

    Минимум -- потому что применимость схемы обязана держаться для ВСЕХ
    тензоров группы, иначе квантование упадёт на середине файла: у cmix
    это key [4D, D] и value [D, 4D], у proj -- четыре разные формы.

    Флаг all_2d НЕ декоративный. Первая версия просто пропускала не-2-D
    тензоры, и для группы `small` возвращала IN=64 по одному лишь r_k
    [12, 64] -- после чего поиск честно предлагал блочную схему, а она
    падала на k_k формы (1,1,768). То есть «взять минимум по тем, что
    подходят» тихо теряло тензоры, которые как раз и определяют
    применимость.
    """
    from .outlier_scan import LORA_BIAS_SUFFIXES
    ins, all_2d = [], True
    for key, t in state_dict.items():
        if getattr(t, "dim", None) is None:
            continue
        if not any(key.endswith(p) or p in key for p in patterns):
            continue
        if key.endswith(LORA_BIAS_SUFFIXES):
            continue           # writer их не квантует -- см. таблицу там же
        if t.dim() < 2:
            continue           # 1-D writer вообще не квантует (bits=16)
        if t.dim() != 2:
            all_2d = False
            continue
        ins.append(int(t.shape[1]))
    if not ins and not all_2d:
        # группа целиком из не-2-D: IN брать неоткуда, но rtn@8 применим
        return 1, False
    return (min(ins) if ins else 0), all_2d


def group_in_features(state_dict, group, patterns):
    """Совместимость: только число каналов (см. group_shapes)."""
    return group_shapes(state_dict, group, patterns)[0]
