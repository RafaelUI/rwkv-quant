"""
Высокоуровневый API. Два входа:
  - quantize(ckpt, out, preset=...)   -- быстрый старт, готовые пресеты
  - quantize(ckpt, out, config=...)   -- полный контроль через QuantConfig
  - calibrate(ckpt, corpus)           -- подобрать QuantConfig под конкретный
                                          чекпоинт вместо пресета "с потолка"
                                          (см. README: чувствительность к
                                          квантованию НЕ переносится между
                                          масштабами модели)
"""
import time

import torch

from .presets import PRESETS
from .calibration import GROUPS, QuantConfig
from .calibration.ablation import perplexity, combined_sanity_check
from .calibration.outlier_scan import GROUP_KEY_PATTERNS
from .calibration import schema_space as _ss
from .models.rwkv7_ref import RWKV7Ref
from .formats import save, quantize_file  # noqa: F401 (save -- публичный API)


def quantize(checkpoint_path: str, output_path: str, preset: str = "reduction",
             config: QuantConfig = None, real_gw: bool = True,
             verbose: bool = True, tokenizer: str = None):
    """
    Quick-start: quantize(ckpt, out, preset="compression")
    Advanced:    quantize(ckpt, out, config=QuantConfig(proj=4, ...))

    preset игнорируется, если передан config.

    real_gw=True (по умолчанию) -- реальная упаковка sb6, файл сжимается.
    real_gw=False -- fake-quant для измерения ppl: та же математика ошибки,
    но веса остаются плотными bf16 и файл НЕ уменьшается.
    """
    if config is None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, choose from {list(PRESETS)}")
        config = PRESETS[preset]

    # Потоковый путь: mmap + потензорное квантование с немедленным
    # освобождением, метаданные -- из форм тензоров. Прежняя версия
    # сначала инстанцировала RWKV7Ref ради пяти чисел (то есть поднимала
    # всю модель в bf16, 5.9 ГБ на 2.9B), потом грузила state_dict ЕЩЁ
    # РАЗ целиком -- на 16 ГБ это давало пик 9-12 ГБ и своп.
    return quantize_file(checkpoint_path, output_path, config,
                         real_gw=real_gw, verbose=verbose, tokenizer=tokenizer)


def _load_corpus(path, device, n_seq=None, seq_len=None):
    """Корпус из .pt: либо тензор [n, T], либо словарь build_eval_multiling."""
    d = torch.load(path)
    tok = d["tokens"] if isinstance(d, dict) else d
    if n_seq:
        tok = tok[:n_seq]
    if seq_len:
        tok = tok[:, :seq_len]
    return tok.contiguous().to(device)


def _mk_config(kw, act_stats_path):
    return QuantConfig(group_scale=dict(kw["group_scale"]),
                       group_scale_mode=dict(kw["group_scale_mode"]),
                       act_stats_path=act_stats_path,
                       **kw["bits"])


def calibrate(checkpoint_path: str, eval_corpus_path: str, device: str = "mps",
              ppl_threshold_pct: float = 5.0, act_stats_path: str = None,
              groups=None, n_seq: int = None, seq_len: int = None,
              verbose: bool = True):
    """Подобрать QuantConfig под конкретный чекпоинт.

    ЧТО ЗДЕСЬ ИСПРАВЛЕНО (10.08.2026) И ПОЧЕМУ ЭТО БЫЛО ВАЖНО.
    Прежняя версия перебирала битность из (8,6,4,2), меряя ppl через
    `fake_quant.q()`, которая знала ТОЛЬКО per-row RTN, и возвращала
    QuantConfig БЕЗ `group_scale`. Отсюда два независимых дефекта:

      - критерий. На 0.1B при bf16 ppl 14.58 группа cmix@4 даёт +1060%
        по per-row и +8.5% по groupwise sb6 -- расхождение в 125 раз.
        При пороге 5% старый поиск отвергал cmix@4 как катастрофу и
        уводил группу на 8/16 бит, то есть раздувал файл, спасаясь от
        деградации, которой в деплое нет;
      - артефакт. Конфиг без `group_scale`, поданный в
        `quantize(config=...)`, уходил в per-row ветку `quantize_tensor`,
        то есть буквально производил ту схему, которую README называет
        сломанной (canonical int4 -> ppl 3798).

    Теперь поиск идёт по ПРИМЕНИМЫМ схемам (calibration/schema_space.py:
    применимость выводится из форм тензоров, а не из таблицы имён),
    измеряется той же функцией, что пишет writer (гейт
    tests/test_calib_matches_writer.py), и минимизирует РЕАЛЬНЫЕ биты на
    вес вместе с накладными раскладки.

    ppl_threshold_pct -- бюджет деградации КОМПОЗИТА, то есть модели
    целиком. Он же используется как отсев на изолированной стадии, но
    там это именно СКРИН, а не гарантия: восемь групп, каждая в пределах
    5%, дают композит заметно выше 5% -- ошибки складываются (закон 5).
    Поэтому после изолированного отбора идёт доводка, которая поднимает
    группы, пока композит не уложится в бюджет, и честно сообщает, если
    не уложился.

    act_stats_path: без него AW-режимы не рассматриваются вовсе, потому
    что без статистики они ВЫРОЖДАЮТСЯ в свои _search-варианты и
    измерялись бы повторно.

    Стоимость: примерно (число групп x число кандидатов) прогонов ppl.
    На 0.1B прогон ~5 с; на 1.5B кратно дороже -- сузьте groups/n_seq.
    """
    groups = list(groups or GROUPS)
    t_start = time.time()

    model = RWKV7Ref(checkpoint_path, device=device, dtype=torch.bfloat16)
    data = _load_corpus(eval_corpus_path, device, n_seq, seq_len)
    n_pred = int(data.shape[0] * (data.shape[1] - 1))

    sd = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    have_act = bool(act_stats_path)

    baseline = perplexity(model, data, QuantConfig())
    if verbose:
        print(f"[calibrate] корпус {tuple(data.shape)} = {n_pred} предсказаний")
        print(f"[calibrate] BASELINE bf16 ppl={baseline:.4f}, "
              f"порог Δ={ppl_threshold_pct:.1f}%, act_stats="
              f"{'есть' if have_act else 'нет (AW-режимы не рассматриваются)'}\n")

    kw = {"bits": {g: 16 for g in GROUPS},
          "group_scale": {}, "group_scale_mode": {}}
    report, n_runs = {}, 0

    for group in groups:
        inf, all_2d = _ss.group_shapes(sd, group, GROUP_KEY_PATTERNS[group])
        if inf == 0:
            if verbose:
                print(f"{group:10s} -- нет квантуемых тензоров, остаётся bf16")
            report[group] = {"chosen": None, "in_features": 0,
                             "all_2d": True, "tried": []}
            continue
        cands = _ss.candidates_for(inf, have_act_stats=have_act, all_2d=all_2d)
        tried, chosen, chosen_i = [], None, None
        for ci, c in enumerate(cands):
            trial = {"bits": dict(kw["bits"]),
                     "group_scale": dict(kw["group_scale"]),
                     "group_scale_mode": dict(kw["group_scale_mode"])}
            # изоляция: все прочие группы -- bf16
            trial["bits"] = {g: 16 for g in GROUPS}
            trial["group_scale"], trial["group_scale_mode"] = {}, {}
            c.apply_to(trial, group)
            t0 = time.time()
            ppl = perplexity(model, data, _mk_config(trial, act_stats_path))
            n_runs += 1
            delta = 100 * (ppl - baseline) / baseline
            tried.append({"cand": repr(c), "ppl": ppl, "delta_pct": delta})
            if verbose:
                mark = "  <- берём" if delta <= ppl_threshold_pct else ""
                print(f"{group:10s} {repr(c):34s} ppl={ppl:11.4f} "
                      f"Δ={delta:+8.2f}%  [{time.time()-t0:4.1f}s]{mark}")
            if delta <= ppl_threshold_pct:
                chosen, chosen_i = c, ci
                break
        if chosen is not None:
            chosen.apply_to(kw, group)
        report[group] = {"chosen": repr(chosen) if chosen else None,
                         "in_features": inf, "all_2d": all_2d, "tried": tried,
                         # индекс, а НЕ битность: rtn@8 и asym@8 совпадают по
                         # битам, и поиск «следующего по цене» по битности
                         # возвращался в ту же точку бесконечно
                         "cand_i": chosen_i, "iso_delta": (
                             tried[chosen_i]["delta_pct"] if chosen_i is not None
                             else 0.0),
                         "cands": [repr(c) for c in cands]}
        if verbose and chosen is None:
            print(f"{group:10s} ни один кандидат не влез в порог -> bf16")

    cfg = _mk_config(kw, act_stats_path)
    if verbose:
        print(f"\n[calibrate] изолированный подбор: {n_runs} прогонов, "
              f"{time.time()-t_start:.0f} с")

    # Изолированный выигрыш НЕ переносится в композит (закон 5): четыре
    # LoRA-ветки на INT4 порознь безобидны, вместе давали ~150x на 1.5B.
    ppl_all = perplexity(model, data, cfg)
    delta_all = 100 * (ppl_all - baseline) / baseline
    if verbose:
        print(f"[calibrate] КОМПОЗИТ: ppl={ppl_all:.4f}  Δ={delta_all:+.2f}%")

    # ---- доводка композита до бюджета ----
    #
    # Изолированный отбор ФИЗИЧЕСКИ не может попасть в бюджет: восемь
    # групп по 5% дают композит заметно выше 5%, ошибки складываются.
    # Прежний код прятал это за множителем ("допуск = порог x 5") и
    # останавливался на числе, которого пользователь не просил.
    #
    # Здесь бюджет означает ровно то, что написано: Δppl КОМПОЗИТА. Пока
    # он превышен, поднимается та группа, у которой хуже всего отношение
    # «изолированная деградация на добавленный бит» -- то есть та, где
    # лишний бит покупает больше всего качества. Прокси по изолированной
    # деградации взят не от лени: leave-one-out по группам
    # (tests/ablate_group_contrib.py) -- та же логика, которой репозиторий
    # уже пользуется, и она стоит одного прогона на шаг вместо восьми.
    # Состояние доводки -- ИНДЕКС кандидата на группу, а не битность.
    # Первая версия искала «текущего» по совпадению bits, и на g_lora
    # застряла: rtn@8 и asym@8 равны по битам, поэтому «следующим по
    # цене» каждый раз оказывался один и тот же asym@8. Одиннадцать
    # прогонов подряд с одинаковым результатом, прежде чем это заметили.
    cur_i = {g: report[g].get("cand_i") for g in groups}
    exhausted = set()
    guard, upgrades = 0, []
    while delta_all > ppl_threshold_pct and guard < 20:
        guard += 1
        best, best_gain, best_next, best_i = None, None, None, None
        for g in groups:
            if g in exhausted or kw["bits"][g] >= 16 or cur_i[g] is None:
                continue
            cands = _ss.candidates_for(
                report[g]["in_features"], have_act_stats=have_act,
                all_2d=report[g].get("all_2d", True))
            i = cur_i[g]
            nxt = cands[i + 1] if i + 1 < len(cands) else None
            add_bits = ((nxt.eff_bits if nxt else 16.0) - cands[i].eff_bits)
            if add_bits <= 0:
                exhausted.add(g)
                continue
            gain = report[g]["iso_delta"] / add_bits
            if best_gain is None or gain > best_gain:
                best, best_gain, best_next, best_i = g, gain, nxt, i + 1
        if best is None:
            break

        prev = (dict(kw["bits"]), dict(kw["group_scale"]),
                dict(kw["group_scale_mode"]))
        if best_next is None:
            kw["bits"][best] = 16
            kw["group_scale"].pop(best, None)
            kw["group_scale_mode"].pop(best, None)
        else:
            best_next.apply_to(kw, best)
        cfg = _mk_config(kw, act_stats_path)
        ppl_new = perplexity(model, data, cfg)
        delta_new = 100 * (ppl_new - baseline) / baseline

        # Подъём обязан УЛУЧШАТЬ композит. Если не улучшил -- откатываем и
        # больше эту группу не трогаем: платить биты за ухудшение бессмысленно,
        # а взаимодействие групп непредсказуемо (закон 5), так что "поднял
        # и стало хуже" -- нормальный исход, а не аномалия.
        if delta_new >= delta_all - 1e-9:
            kw["bits"], kw["group_scale"], kw["group_scale_mode"] = prev
            exhausted.add(best)
            cfg = _mk_config(kw, act_stats_path)
            if verbose:
                print(f"[calibrate] {best} -> {best_next or 'bf16'} не помог "
                      f"({delta_new:+.2f}% против {delta_all:+.2f}%), откат")
            continue
        cur_i[best] = best_i if best_next is not None else None
        ppl_all, delta_all = ppl_new, delta_new
        upgrades.append({"group": best, "to": repr(best_next) if best_next else "bf16",
                         "delta_pct": delta_all})
        if verbose:
            print(f"[calibrate] поднимаю {best} -> {best_next or 'bf16'}: "
                  f"композит Δ={delta_all:+.2f}%")

    if delta_all > ppl_threshold_pct and verbose:
        print(f"[calibrate] ВНИМАНИЕ: бюджет {ppl_threshold_pct:.1f}% не достигнут "
              f"за {guard} шагов (сейчас {delta_all:+.2f}%). Либо бюджет слишком "
              f"жёсткий для этого чекпоинта, либо расширьте пространство схем.")

    if verbose:
        print(f"\n{cfg}")
        print(f"group_scale={cfg.group_scale}\ngroup_scale_mode={cfg.group_scale_mode}")
        print(f"[calibrate] итог Δppl(композит) = {delta_all:+.2f}%, "
              f"всего {time.time()-t_start:.0f} с")
    cfg.calibration_report = {
        "checkpoint": checkpoint_path, "baseline_ppl": baseline,
        "n_pred": n_pred, "threshold_pct": ppl_threshold_pct,
        "combined_ppl": ppl_all, "combined_delta_pct": delta_all,
        "groups": report, "upgrades": upgrades,
    }
    return cfg
