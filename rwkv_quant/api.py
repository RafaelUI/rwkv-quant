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
import copy
import os
import time

import torch

from .presets import PRESETS
from .calibration import GROUPS, QuantConfig
from .calibration.ablation import perplexity, combined_sanity_check
from .calibration.outlier_scan import GROUP_KEY_PATTERNS
from .calibration import schema_space as _ss
from .models.rwkv7_ref import RWKV7Ref
from .formats import save, quantize_file  # noqa: F401 (save -- публичный API)
from .calibration import act_stats as act_stats_mod
def quantize(checkpoint_path: str, output_path: str, preset: str = "reduction",
             config: QuantConfig = None, real_gw: bool = True,
             verbose: bool = True, tokenizer=None, act_stats="auto",
             autopick=None, autopick_budget: float = 0.005,
             measure="auto", device: str = None, gptq=None, gptq_calib=None,
             allow_per_row: bool = False):
    """
    Quick-start: quantize(ckpt, out, tokenizer=tok, preset="compression")
    Advanced:    cfg = copy.deepcopy(rwkv_quant.presets.COMPRESSION); cfg.bits["proj"] = 5
                 quantize(ckpt, out, tokenizer=tok, config=cfg)

    preset игнорируется, если передан config.

    ПРОВЕРКИ ДО НАЧАЛА РАБОТЫ (06.10; гейт tests/test_api_misuse.py). Отказ сразу, а не после
    часов счёта и не молча: каталога выходного файла нет; output_path -- сам чекпоинт (затёрся
    бы); чекпоинт не читается или не RWKV-7; config не QuantConfig; значения QuantConfig вне
    допустимого (биты -- целое 1..8 или 16); шаблон bits_overrides не совпал ни с одной
    матрицей (шаблоны самих пресетов под другое именование ключей не считаются); явный
    act_stats не покрывает AW-матрицы этого чекпоинта; gptq_calib с токенами вне словаря.
    gptq=True на конфиге без целей GPTQ (нет proj / cmix / head в asym_sb6* или sym*) --
    предупреждение и пропуск вместо полного прохода.

    СВОЙ config -- ОТ ПРЕСЕТА, А НЕ С НУЛЯ. QuantConfig(proj=4, cmix=4, ...) "только биты" не
    задаёт group_scale, и такие группы уходят в построчный RTN. Ниже 8 бит это либо сломанный
    файл (0.1B: cmix=4 построчно -- KL 2.8, прежний пример отсюда давал 3.1), либо пустая трата
    (коды 5-7 бит лежат целым байтом: размер как у 8 бит, качество хуже). Поэтому quantize()
    ОТКАЗЫВАЕТ (ValueError до любой работы), если конфиг отправляет хоть одну матрицу в
    построчный путь ниже 8 бит. Варианты: копия пресета с правкой bits (как выше); свои
    group_scale / group_scale_mode; 8 бит; или allow_per_row=True -- осознанно, для
    исследований (поведение и байты прежние). calibrate() таких конфигов не выдаёт:
    построчных 4 бит среди его кандидатов нет (06.10), исключений из отказа нет.
    Там же отказ на неизвестное имя группы (QuantConfig(prooj=4) раньше молча давал файл
    целиком в bf16) и на group_scale_mode без group_scale (режим молча игнорировался) --
    QuantConfig.validate(), зовётся и конструктором.

    real_gw=True (по умолчанию) -- реальная упаковка sb6, файл сжимается.
    real_gw=False -- fake-quant для измерения ppl: та же математика ошибки,
    но веса остаются плотными bf16 и файл НЕ уменьшается.

    tokenizer -- объект с .encode, callable или путь к словарю. Нужен не
    для метаданных, а для КАЛИБРОВКИ: пресеты взвешивают ошибку по
    активности входных каналов, а чтобы её измерить, репозиторный корпус
    (rwkv_quant/data/calib_corpus.txt) надо разобрать ТЕМ ЖЕ словарём, что
    у чекпоинта. Путь понимается как словарь RWKV World (rwkv_vocab_*.txt): на Mac его
    разбирает токенизатор rwkv-metal, на прочих системах -- его копия в пакете
    (rwkv_quant/world_tokenizer.py). Словарь не от этого чекпоинта ловится только по id за
    пределами vocab; чужой словарь с id в пределах проходит и молча портит статистику.

    act_stats:
      "auto" (умолчание) -- снять статистику самим и закешировать в
        ~/.cache/rwkv-quant (ключ -- чекпоинт и ТОКЕНЫ калибровочных окон: один словарь в
        любой форме -- одна запись, разные словари -- разные; 06.10); занимает секунды, но требует прямого прохода
        по ПЛОТНОЙ модели (~3 ГБ на 1.5B, ~6 ГБ на 2.9B). Шаг идёт ДО
        квантования и память освобождается, поэтому пик процесса --
        максимум из двух, а не сумма;
      путь -- взять готовый файл (проверка принадлежности чекпоинту
        остаётся на вызывающем);
      None -- ОСОЗНАННО без AW. Измеренная цена: KL(bf16 || квант) хуже на
        38%, top-1 на 0.78 п.п. (NEXT_SESSION, раздел 9).

    autopick (24.09): None (умолчание) -- ВКЛЮЧЁН для preset="compression" без своего
      config (решение владельца 24.09; проверен только там), иначе выключен. autopick=True на
      раскладке не sb6 (preset="reduction", построчные группы) -- ValueError до измерения
      (06.10; раньше падало NotImplementedError после него). При неявном
      включении пропускается с предупреждением, если плотная bf16-модель не влезает в
      половину памяти устройства (измерению нужна плотная модель). True/False -- явно.
      Суть: перераспределить биты по матрицам по
      измерению чувствительности (calibration.autopick): подъёмы, где байт окупается
      сильнее, спуски (>= 4 бит), где слабее, в пределах бюджета autopick_budget --
      ДОЛИ ФАЙЛА (0.005 = +0.5%; 0 -- байт-нейтрально). g1j 1.5B, +0.5%: KL на
      отложенном тексте -16%, Δppl +4.15% -> +3.13%, на отложенном коде -13.5%.
      Цель -- широкая квота окон с весами пропорционально окнам (autopick.QUOTA).
    measure: "auto" -- измерить здесь (кеш ~/.cache/rwkv-quant/measure; ~1 ч на 1.5B
      на M4, линейно по числу окон; нужна плотная модель в bf16 на device) или путь к
      JSON готового измерения (перенос с машины с большей памятью, как imatrix): чужой
      чекпоинт -- отказ, иная подпись конфига/корпуса -- предупреждение.
    device: для измерения и GPTQ ("mps"/"cuda"/"cpu" или список карт "cuda:0,cuda:1" -- слои по картам;
      по умолчанию RWKVQ_DEVICE или mps/cpu).

    gptq (30.09): None (умолчание) -- ВКЛЮЧЁН для preset="compression" (решение владельца 30.09) и
      preset="reduction" (01.10: KL к RTN -21...-30% на 0.1B-7.2B, значимо на каждом языке; ppl в шуме --
      выигрыш в верности bf16, важной для QLoRA-базы и векторных моделей) без своего config, иначе выключен; при неявном включении пропускается с предупреждением, если
      плотная модель и активации калибровки не влезают (см. _gptq_skip_reason) или нет корпуса.
      True/False -- явно. Суть: коды матриц proj/cmix/head, которые пресет пишет в sb6, считаются
      GPTQ (компенсация ошибки округления через H^-1, calibration.gptq) на ТОЙ ЖЕ сетке -- формат и
      кернели те же, размер файла тот же. Калибровка -- 600 окон (решение владельца: всегда), damp 0.1.
      Цифры (KL к RTN, отложенный текст / код, 600 окон): 1.5B -42.2 / -49.9%, 2.9B -42.5 / -46.4%,
      7.2B -28.8 / -38.5%, 13.3B -28.8 / -38.5%; вместе с autopick +0.5% к прежнему пути: 7.2B
      -42.5 / -44.9%, 13.3B -41.3 / -47.6%. Цена -- время: 13.3B ~4.7 ч на 4x4090.
    gptq_calib: None -- корпус пакета; путь к .pt ({"tokens": [N, T]}) или тензор [N, T] токенов.
        Свой набор -- НЕ МЕНЬШЕ 600 окон: берутся первые 600 окон и первые 512 токенов каждого,
        на меньшем наборе ValueError (число окон фиксировано решением владельца).
    """
    _preflight_args(checkpoint_path, output_path, config, act_stats, device)
    _user_config = config is not None
    if config is None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, choose from {list(PRESETS)}")
        # КОПИЯ, а не сам пресет: ниже проставляется act_stats_path, а
        # PRESETS -- разделяемые объекты уровня модуля. Мутация утекла бы
        # в следующий вызов quantize() в том же процессе.
        config = copy.deepcopy(PRESETS[preset])
    else:
        config = copy.deepcopy(config)

    _validate_config(config)
    if not allow_per_row:
        _refuse_per_row(checkpoint_path, config)
    if autopick:
        _refuse_autopick(config)
    _refuse_unmatched_overrides(checkpoint_path, config)

    calib_sig = None
    needs_aw = any(str(m).endswith("_aw")
                   for m in (config.group_scale_mode or {}).values())
    if act_stats == "auto":
        if needs_aw:
            stats, calib_sig = act_stats_mod.collect(
                checkpoint_path, tokenizer, verbose=verbose)
            config.act_stats_path = os.path.join(
                act_stats_mod.CACHE_DIR, "act_%s.pt" % calib_sig)
            del stats
        else:
            config.act_stats_path = None
    elif act_stats is None:
        # Отказ ОСОЗНАННЫЙ и видимый: тихое вырождение AW -- это ровно та
        # ошибка, из-за которой сбор и переехал в библиотеку.
        if needs_aw and verbose:
            print("[act_stats] ОТКЛЮЧЕНО вызывающим: AW-режимы вырождаются "
                  "в невзвешенный поиск (измеренная цена: KL +38%)")
        config.act_stats_path = None
    else:
        if not os.path.exists(act_stats):
            raise FileNotFoundError("act_stats=%r не существует" % act_stats)
        if needs_aw:
            _check_act_stats(act_stats, checkpoint_path, config)
        config.act_stats_path = act_stats

    ap_meta = None
    implicit = autopick is None
    if implicit:
        autopick = (preset == "compression" and not _user_config)
    if autopick and implicit and not _fits_for_measure(checkpoint_path, device):
        autopick = False
        if verbose:
            print("[autopick] пропущен: плотная bf16-модель больше половины памяти устройства; "
                  "measure -- на машине с большей памятью, затем quantize(measure=путь)")
    if autopick:
        _refuse_autopick(config)
        ap_meta = _autopick(checkpoint_path, config, tokenizer, autopick_budget,
                            measure, device, verbose)

    gptq_q = gptq_meta = None
    implicit_g = gptq is None
    if implicit_g:
        gptq = (preset in ("compression", "reduction") and not _user_config)   # reduction -- решение владельца 01.10
    if gptq and not _gptq_has_targets(config):
        # 06.10: раньше шёл полный проход (16 минут на 0.1B на Mac) и только потом сообщал,
        # что менять нечего
        if not implicit_g:
            import warnings
            warnings.warn("gptq=True пропущен: в конфиге нет групп proj / cmix / head с group_scale в "
                          "режимах asym_sb6* или sym*, а другие GPTQ не обрабатывает", stacklevel=2)
        gptq = False
    if gptq and not real_gw:
        if not implicit_g:
            raise ValueError("gptq=True пишет коды sb6; с real_gw=False не сочетается")
        gptq = False
    if gptq and implicit_g:
        why = _gptq_skip_reason(checkpoint_path, device, gptq_calib)
        if why:
            gptq = False
            if verbose:
                print("[gptq] пропущен: %s; явно -- quantize(gptq=True), на машине с большей памятью" % why)
    if gptq:
        gptq_q, gptq_meta = _gptq(checkpoint_path, config, tokenizer, gptq_calib, device, verbose)

    # В манифест едет ОПИСАНИЕ калибровки, а не только имя словаря: файл
    # должен сам отвечать на вопрос "чем это калибровалось".
    tok_label = tokenizer if isinstance(tokenizer, str) else (
        type(tokenizer).__name__ if tokenizer is not None else None)
    if calib_sig:
        tok_label = "%s (calib %s)" % (tok_label or "?", calib_sig)

    # Потоковый путь: mmap + потензорное квантование с немедленным
    # освобождением, метаданные -- из форм тензоров. Прежняя версия
    # сначала инстанцировала RWKV7Ref ради пяти чисел (то есть поднимала
    # всю модель в bf16, 5.9 ГБ на 2.9B), потом грузила state_dict ЕЩЁ
    # РАЗ целиком -- на 16 ГБ это давало пик 9-12 ГБ и своп.
    return quantize_file(checkpoint_path, output_path, config,
                         real_gw=real_gw, verbose=verbose, tokenizer=tok_label,
                         autopick=ap_meta, gptq_tensors=gptq_q, gptq_meta=gptq_meta)


# Группы, которым autopick переназначает биты (calibration.autopick.R_LEVEL), и раскладка, под
# которую он сделан: лестница 4 / 5 / 6 бит -> bf16 в режимах asym_sb6*, байты (b + 0.5) / 8.
_AUTOPICK_GROUPS = ("proj", "cmix", "emb", "head")
_AUTOPICK_BITS = (4, 5, 6)


def _autopick_unsupported(config):
    """[(группа, режим, биты)] -- группы, на которых autopick работать не умеет."""
    bad = []
    for g in _AUTOPICK_GROUPS:
        b = config.bits.get(g, 16)
        if b >= 16:
            continue
        mode = (config.group_scale_mode or {}).get(g)
        if not (config.group_scale or {}).get(g) or not str(mode).startswith("asym_sb6") or b not in _AUTOPICK_BITS:
            bad.append((g, mode, b))
    return bad


def _refuse_autopick(config):
    """Отказ ДО измерения (06.10, решение владельца). autopick назначает матрицам биты по лестнице
    sb6 (4 / 5 / 6 -> bf16). На другой раскладке он выдаёт нереализуемые точки: preset="reduction"
    (sym_aw, 8 бит) + autopick=True падал NotImplementedError «mode=sym_aw bits=7» уже ПОСЛЕ
    измерения (час на 1.5B). Поддержка autopick для reduction -- отдельное необязательное
    исследование (NEXT_SESSION, 06.10). Гейт: tests/test_autopick_refusal.py."""
    bad = _autopick_unsupported(config)
    if bad:
        raise ValueError(
            "autopick=True не поддерживается для этого конфига: он переназначает биты по лестнице "
            "sb6 (режимы asym_sb6*, 4-6 бит), а здесь %s. Автоподбор сделан и проверен для "
            "preset=\"compression\" (там он включён по умолчанию); для preset=\"reduction\" и своих "
            "раскладок уберите autopick=True."
            % ", ".join("%s: режим %s, %d бит" % (g, m or "построчный", b) for g, m, b in bad))


def _preflight_args(ckpt, out, config, act_stats, device):
    """Проверки аргументов ДО любой работы (06.10, проверка «неверным использованием»): то, что
    раньше всплывало невнятной ошибкой из недр или -- хуже -- после часов счёта."""
    for name, v in (("checkpoint_path", ckpt), ("output_path", out)):
        if not isinstance(v, (str, os.PathLike)):
            raise TypeError("%s должен быть путём (str), а получен %s" % (name, type(v).__name__))
    if config is not None and not isinstance(config, QuantConfig):
        raise TypeError(
            "config должен быть QuantConfig, а получен %s. Имя пресета передаётся как "
            "preset=\"compression\"; свой конфиг -- QuantConfig(...) или копия "
            "rwkv_quant.presets.COMPRESSION." % type(config).__name__)
    if device is not None and not isinstance(device, (str, list, tuple)):
        raise TypeError("device должен быть строкой (\"mps\", \"cuda:0\", \"cuda:0,cuda:1\") "
                        "или списком таких строк, а получен %s" % type(device).__name__)
    if not (act_stats is None or isinstance(act_stats, (str, os.PathLike))):
        raise TypeError("act_stats: \"auto\", None или путь к файлу статистики, а получен %s"
                        % type(act_stats).__name__)
    ckpt, out = os.fspath(ckpt), os.fspath(out)
    if not os.path.exists(ckpt):
        raise FileNotFoundError("чекпоинт не найден: %s" % ckpt)
    outdir = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(outdir):
        raise FileNotFoundError(
            "каталога для выходного файла нет: %s (проверено до начала работы -- создайте его)" % outdir)
    if not os.access(outdir, os.W_OK):
        raise PermissionError("в каталог выходного файла нельзя писать: %s" % outdir)
    src = os.path.join(ckpt, "model.safetensors") if os.path.isdir(ckpt) else ckpt
    if os.path.exists(out) and os.path.exists(src) and os.path.samefile(src, out):
        raise ValueError("output_path совпадает с чекпоинтом (%s): квантованный файл затёр бы "
                         "исходную модель. Укажите другой путь." % out)
    from .formats import writer as _w
    try:
        sd = _w._open_sd(ckpt)
    except Exception as e:
        raise ValueError(
            "не удалось прочитать чекпоинт %s: ожидается .pth (torch) либо файл или каталог "
            "safetensors. (%s: %s)" % (ckpt, type(e).__name__, str(e).split("\n")[0][:120])) from e
    try:
        _w.detect_meta(ckpt, sd)
    except Exception as e:
        raise ValueError(
            "%s не похож на чекпоинт RWKV-7: по именам и формам тензоров не удалось определить "
            "число слоёв и размерности. (%s: %s)" % (ckpt, type(e).__name__, str(e)[:80])) from e


def _refuse_unmatched_overrides(ckpt, config):
    from .formats import writer as _w
    # Пресеты несут шаблоны под ОБА именования ключей (world и custom): на любом чекпоинте часть
    # их шаблонов законно не совпадает. Опечатку от чужого именования по шаблону не отличить,
    # поэтому шаблоны самих пресетов из проверки исключены.
    known = {p for c in PRESETS.values() for p in (getattr(c, "bits_overrides", None) or {})}
    bad = [p for p in _w.unmatched_overrides(ckpt, config) if p not in known]
    if bad:
        raise ValueError(
            "bits_overrides: шаблон не совпал ни с одной квантуемой матрицей чекпоинта: %s. "
            "Шаблон -- подстрока имени тензора (например \"blocks.3.att.key.weight\"); "
            "несовпавший шаблон раньше молча ничего не делал."
            % ", ".join(repr(p) for p in bad[:8]))


def _check_act_stats(path, ckpt, config):
    """Явно переданный файл статистики обязан покрывать AW-матрицы ЭТОГО чекпоинта: иначе
    AW-режимы молча вырождаются в невзвешенные (groupwise.get_ex2 возвращает None), а файл
    пишется как ни в чём не бывало."""
    from .formats import writer as _w
    from .calibration import groupwise as _gw
    try:
        stats = _gw.load_act_stats(path)
        ok = isinstance(stats, dict)
    except Exception as e:
        raise ValueError("act_stats=%r не читается как файл статистики активаций (%s: %s)"
                         % (path, type(e).__name__, str(e)[:100])) from e
    aw = {g for g, m in (config.group_scale_mode or {}).items() if str(m).endswith("_aw")} - {"emb"}
    need, miss = 0, []
    for key, w in _w._open_sd(ckpt).items():
        g = _w._match_group(key)
        if g in aw and _w._is_quantized(key, g, w.dim()):
            need += 1
            if not ok or _gw.get_ex2(path, key, w) is None:
                miss.append(key)
    if miss:
        raise ValueError(
            "act_stats=%r не подходит к этому чекпоинту: статистики нет (или она другой ширины) "
            "для %d из %d матриц AW-групп, например %s. Файл от другой модели или не того "
            "формата; с act_stats=\"auto\" библиотека соберёт статистику сама."
            % (path, len(miss), need, miss[0]))


def _gptq_has_targets(config):
    """Может ли GPTQ хоть что-то сделать с конфигом: группа proj / cmix / head с group_scale в
    режиме sb6 (compression) или sym* (reduction) -- по тому же признаку, что gptq.run.plan.
    Битность здесь НЕ проверяется (её правят bits_overrides): False -- только когда наверняка нечего."""
    from .calibration import gptq as G
    def ok(g):
        mode = str((config.group_scale_mode or {}).get(g, "asym"))
        return bool((config.group_scale or {}).get(g)) and (mode in G._SB_BITS or mode.startswith("sym"))
    return any(ok(g) for g in ("proj", "cmix", "head"))


def _validate_config(config):
    """Итоговый конфиг (после копии): неизвестные группы и режим без group_scale -- отказ.
    Конструктор QuantConfig проверяет то же, но поля -- словари, их правят и после сборки."""
    config.validate()


def _refuse_per_row(ckpt, config):
    """Отказ ДО любой работы, если config шлёт матрицы в построчный RTN ниже 8 бит (06.10,
    решение владельца: отказ, а не предупреждение). Гейт: tests/test_per_row_refusal.py."""
    from .formats import writer as _w
    bad = _w.per_row_low_bits(ckpt, config)
    if not bad:
        return
    by = {}
    for key, group, bits in bad:
        by.setdefault((group, bits), []).append(key)
    what = ", ".join("%s=%d (%d матриц, напр. %s)" % (g, b, len(ks), ks[0])
                     for (g, b), ks in sorted(by.items()))
    raise ValueError(
        "конфиг отправляет матрицы в построчный RTN ниже %d бит: %s.\n"
        "У этих групп нет group_scale, а построчная схема ниже %d бит либо ломает модель "
        "(4 бита: KL к bf16 0.2-2.8 на группу), либо ничего не экономит (5-7 бит хранятся "
        "целым байтом -- размер как у 8).\n"
        "  * от пресета: cfg = copy.deepcopy(rwkv_quant.presets.COMPRESSION); cfg.bits[\"proj\"] = 5;\n"
        "  * или задайте group_scale / group_scale_mode для этих групп (см. presets.py);\n"
        "  * или поднимите их до 8 бит;\n"
        "  * осознанно построчно: quantize(..., allow_per_row=True)."
        % (_w.PER_ROW_MIN_BITS, what, _w.PER_ROW_MIN_BITS))


# Доля памяти под плотную модель + активации калибровки GPTQ при НЕЯВНОМ включении. Активации -- два тензора
# [окна, 511, n_embd] fp32 на хосте (x и v_first). На M4 16 ГБ: 0.1B-1.5B проходят, 2.9B -- нет (решение
# владельца 28.09: квантование 2.9B+ -- дело мощной машины).
GPTQ_FIT_FRACTION = 0.6


def _gptq_skip_reason(ckpt, device, calib):
    """None -- можно; иначе причина пропуска неявного GPTQ."""
    from .calibration import gptq as G
    if calib is None and not os.path.exists(G.CALIB_FILE):
        return "нет корпуса GPTQ пакета (%s)" % os.path.basename(G.CALIB_FILE)
    try:
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
        C = int(sd["emb.weight"].shape[1])
        model = os.path.getsize(ckpt)
        del sd
    except Exception:
        return None
    act = 2 * G.N_WINDOWS * (act_stats_mod.SEQ_LEN - 1) * C * 4
    phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    devs = G.devices(device)
    if devs[0].startswith("cuda") and torch.cuda.is_available():
        card = torch.cuda.get_device_properties(devs[0]).total_memory
        if model / len(devs) > GPTQ_FIT_FRACTION * card:
            return "модель %.1f ГБ на %d карт(ы) по %.0f ГБ" % (model / 1e9, len(devs), card / 1e9)
        if act > GPTQ_FIT_FRACTION * phys:
            return "активации калибровки %.1f ГБ при памяти хоста %.0f ГБ" % (act / 1e9, phys / 1e9)
        return None
    if model + act > GPTQ_FIT_FRACTION * phys:
        return "модель %.1f ГБ + активации калибровки %.1f ГБ больше %.0f%% памяти (%.0f ГБ)" % (
            model / 1e9, act / 1e9, 100 * GPTQ_FIT_FRACTION, phys / 1e9)
    return None


def _gptq(ckpt, config, tokenizer, calib, device, verbose):
    """GPTQ по итоговому config (после autopick) -> (упакованные тензоры, метаданные для манифеста)."""
    import hashlib
    from .calibration import gptq as G
    if any(str(m).endswith("_aw") for m in (config.group_scale_mode or {}).values()) and not config.act_stats_path:
        raise ValueError("gptq с AW-пресетом требует act_stats (сетка писателя строится по той же статистике)")
    tok, label = G.calib_tokens(tokenizer, calib, G.N_WINDOWS)
    from .formats import writer as _w
    vocab = _w.detect_meta(ckpt, _w._open_sd(ckpt))["vocab_size"]
    lo, hi = int(tok.min()), int(tok.max())
    if lo < 0 or hi >= vocab:
        # у act_stats такая проверка была, у GPTQ нет: на MPS выход за vocab молча читал мусор
        # (файл записывался), на CUDA сыпал аппаратными ошибками
        raise ValueError("калибровка GPTQ: токены вне словаря чекпоинта (id от %d до %d при vocab %d) "
                         "-- окна разобраны не тем словарём" % (lo, hi, vocab))
    qts = G.run(ckpt, tokenizer, config, n_windows=G.N_WINDOWS, damp=G.DAMP, device=device,
                verbose=verbose, calib=tok)
    if not qts:
        if verbose:
            print("[gptq] ни одна матрица не идёт в sb6 -- GPTQ ничего не изменил, в манифест не пишется")
        return None, None
    meta = dict(windows=int(tok.shape[0]), seq_len=int(tok.shape[1]), damp=G.DAMP, calib=label,
                calib_sha=hashlib.sha1(tok.numpy().tobytes()).hexdigest()[:16], matrices=len(qts))
    return (qts or None), meta


# Доля памяти устройства под плотную bf16-модель при НЕЯВНОМ autopick. 0.3 осторожно: 1.5B
# (3 ГБ) на 16 ГБ идёт со свопом +0.4 ГБ за measure (24.09); 2.9B (5.8 ГБ) на 16 ГБ не
# проверялся -- неявно не запускается, явный autopick=True -- на ответственности вызывающего.
FIT_FRACTION = 0.3


def _fits_for_measure(ckpt, device):
    """Грубо: bf16-копия (2 байта на параметр ~ размер bf16/fp16 .pth) <= FIT_FRACTION памяти
    устройства (cuda -- память карты, иначе -- физическая память)."""
    try:
        need = os.path.getsize(ckpt) if not os.path.isdir(ckpt) else sum(
            os.path.getsize(os.path.join(ckpt, f)) for f in os.listdir(ckpt))
        dev = device or os.environ.get("RWKVQ_DEVICE", "")
        if dev.startswith("cuda") and torch.cuda.is_available():
            have = torch.cuda.get_device_properties(0).total_memory
        else:
            have = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return need <= FIT_FRACTION * have
    except (OSError, ValueError, AttributeError):
        return True


def _autopick(ckpt, config, tokenizer, budget, measure, device, verbose):
    """Измерение (или перенесённое) -> select_budget -> bits_overrides ПЕРВЫМИ в config
    (подстрока, первое побеждает). Мутирует config (это уже копия). -> метаданные."""
    import json
    from .calibration import autopick as ap
    needs_aw = any(str(m).endswith("_aw") for m in (config.group_scale_mode or {}).values())
    if needs_aw and not config.act_stats_path:
        raise ValueError("autopick с AW-пресетом требует act_stats (измерение без статистики "
                         "мерило бы вырожденный квантователь)")
    if measure == "auto":
        m = ap.measure(ckpt, config, tokenizer, device=device, verbose=verbose)
    else:
        m = json.load(open(measure))
        if m.get("ckpt_sig") != ap._ckpt_signature(ckpt):
            raise ValueError("измерение %s -- от другого чекпоинта (ckpt_sig %s)"
                             % (measure, m.get("ckpt_sig")))
        from .calibration import act_stats as A
        exp = ap._signature(ckpt, config, A.CORPUS, m.get("n_windows", ap.N_WINDOWS),
                            m.get("seq_len", ap.SEQ_LEN), tokenizer)
        if m.get("signature") != exp and verbose:
            print("[autopick] ВНИМАНИЕ: подпись перенесённого измерения %s не равна здешней %s "
                  "(другие AW-статистика/конфиг/корпус/квота или устройство сборки статистики; "
                  "измерения, снятые до 06.10, подписаны по-старому и не совпадут никогда) "
                  "-- выбор берётся как есть" % (m.get("signature"), exp))
    ovr, rep = ap.select_budget(m, budget)
    if not rep["feasible"] and verbose:
        print("[autopick] бюджет %+.2f%% недостижим: взят самый дешёвый выбор (%+.2f%%)"
              % (100 * budget, 100 * rep["bytes_frac"]))
    config.bits_overrides = dict(ovr, **(config.bits_overrides or {}))
    if verbose:
        print("[autopick] бюджет %+.2f%%: tau %.3f, вверх %d, вниз %d, байты %+.2f МБ (%+.3f%%), "
              "предсказание KL %+.1f%%" % (100 * budget, rep["tau"], rep["n_up"], rep["n_down"],
                                          rep["bytes"] / 1e6, 100 * rep["bytes_frac"], 100 * rep["kl_pred"]))
    return dict(measure_signature=m.get("signature"), measure_version=m.get("version"),
                measure_device=m.get("device"), langs=m.get("langs"), budget=budget,
                tau=rep["tau"], feasible=rep["feasible"], n_up=rep["n_up"], n_down=rep["n_down"],
                bytes_frac=rep["bytes_frac"], kl_pred=rep["kl_pred"], overrides=ovr)


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
