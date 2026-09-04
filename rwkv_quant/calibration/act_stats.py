# -*- coding: utf-8 -*-
"""Сбор E[x^2] по входным каналам квантуемых матриц -- ВНУТРИ БИБЛИОТЕКИ.

ЗАЧЕМ ЭТОТ МОДУЛЬ СУЩЕСТВУЕТ. Пресеты используют AW-варианты поиска
масштаба, а те взвешивают ошибку по активности входных каналов. Раньше
статистика бралась из файла, путь к которому был ЗАШИТ В ПРЕСЕТ
(`/tmp/act_stats_1p5b.pt`), и если файла не было -- AW молча вырождался.
Измерено, чем это молчание оборачивается (NEXT_SESSION, разделы 9-12):

  * без статистики KL(bf16 || квант) хуже на 38% (0.004021 против
    0.002908 нат/токен), top-1 падает на 0.78 п.п.;
  * статистика насыщается примерно на ТЫСЯЧЕ токенов -- 2, 4 и 17
    последовательностей дают одно и то же в пределах 2%;
  * а вот ПИСЬМЕННОСТИ важны: на китайском и коде статистика, снятая с
    ru/en/sr, даёт 0.004096 против 0.003783 БЕЗ всякой статистики, то
    есть весь выигрыш AW исчезает начисто.

Отсюда устройство: корпус ЕДЕТ В РЕПОЗИТОРИИ (широкий по языкам, мелкий
по объёму), сбор идёт автоматически при квантовании, а токенизатор
передаёт пользователь -- он привязан к модели, а не к нам.

ЦЕНА, КОТОРУЮ НАДО ЗНАТЬ: сбор требует прямого прохода по ПЛОТНОЙ модели
(bf16 на CPU) -- около 3 ГБ для 1.5B и 6 ГБ для 2.9B. Шаг идёт ДО
квантования и память освобождается, так что пик процесса -- максимум из
двух, а не сумма.
"""
import hashlib
import os
import re
import time

import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")
CORPUS = os.path.join(DATA_DIR, "calib_corpus.txt")
SEQ_LEN = 512
# Бюджет с запасом к точке насыщения (~1000 токенов): берём около 8k, но
# РАСПРЕДЕЛЁННЫХ по всем чанкам, а не подряд -- см. _windows.
TOKEN_BUDGET = 8192
CACHE_DIR = os.path.expanduser("~/.cache/rwkv-quant/act_stats")


class TokenizerRequired(RuntimeError):
    """Токенизатор не передан или не пригоден -- сказать это ГРОМКО.

    Тихая альтернатива (собрать без калибровки) уже измерена и стоит 38%
    KL, поэтому она не предлагается по умолчанию: чтобы отказаться от AW,
    надо написать act_stats=None явно."""


def _encoder(tokenizer):
    """Callable, объект с .encode или путь к словарю -> функция text->ids."""
    if tokenizer is None:
        raise TokenizerRequired(
            "нужен токенизатор: он привязан к чекпоинту, а не к библиотеке.\n"
            "  quantize(..., tokenizer=<объект с .encode>) -- обычный путь;\n"
            "  quantize(..., tokenizer=<callable>) -- если он у вас функция;\n"
            "  quantize(..., act_stats=None) -- ОСОЗНАННО без AW "
            "(измеренная цена: KL хуже на 38%)")
    if callable(tokenizer) and not hasattr(tokenizer, "encode"):
        return tokenizer
    # ПОРЯДОК ПРИНЦИПИАЛЕН: у str метод .encode ЕСТЬ (это кодирование
    # текста в байты), поэтому путь к словарю обязан разбираться ДО
    # проверки "объект с .encode". Иначе строка уходит в str.encode, а
    # чанк корпуса попадает туда как ИМЯ КОДИРОВКИ, и наружу приходит
    # LookupError с текстом чанка вместо внятного отказа.
    if isinstance(tokenizer, (str, os.PathLike)):
        # Путь к словарю. Своего токенизатора у rwkv-quant НЕТ и заводить
        # его -- значит держать вторую копию чужой реализации (закон 23);
        # поэтому пробуем соседний пакет и, если его нет, честно говорим.
        try:
            from rwkv_metal.tokenizer.world_tokenizer import WorldTokenizer
        except ImportError as e:
            raise TokenizerRequired(
                "передан путь к словарю (%s), но разобрать его нечем: "
                "rwkv-quant не везёт своего токенизатора. Передайте готовый "
                "объект с .encode." % tokenizer) from e
        return WorldTokenizer(os.fspath(tokenizer)).encode
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode
    raise TokenizerRequired("не понимаю tokenizer=%r" % (tokenizer,))


def _windows(chunks, encode, seq_len=SEQ_LEN, budget=TOKEN_BUDGET):
    """Окна по seq_len, отобранные ПО КРУГУ между чанками.

    Порядок обхода принципиален. Корпус упорядочен по языкам, и если
    резать подряд до бюджета, последние письменности (в нашем корпусе --
    китайский) не попадут в статистику вовсе. Измерено, что это стоит
    всего выигрыша AW на этих языках, поэтому обход круговой: сначала по
    одному окну с каждого чанка, потом по второму и так далее."""
    per_chunk = []
    for ch in chunks:
        ids = encode(ch)
        per_chunk.append([ids[w * seq_len:(w + 1) * seq_len]
                          for w in range(len(ids) // seq_len)])
    # ПЕРВЫЙ КРУГ БЮДЖЕТУ НЕ ПОДЧИНЯЕТСЯ. Это не мелочь: при бюджете 8192
    # и окне 512 круг обрывается на шестнадцатом чанке, а код и китайский
    # в корпусе стоят двадцать первым и дальше -- то есть "круговой обход"
    # терял их ровно так же, как последовательная нарезка (поймано
    # гейтом tests/test_act_stats_auto.py). Сначала гарантируем по одному
    # окну с КАЖДОГО чанка, и только потом добираем до бюджета.
    out = [wins[0] for wins in per_chunk if wins]
    depth = 1
    while any(depth < len(w) for w in per_chunk):
        if len(out) * seq_len >= budget:
            break
        for wins in per_chunk:
            if depth < len(wins):
                out.append(wins[depth])
                if len(out) * seq_len >= budget:
                    break
        depth += 1
    return out


def _signature(ckpt_path, tokenizer, corpus_path):
    """Подпись входов сбора -- ключ кеша И запись в манифест.

    Чекпоинт хешируется не целиком (гигабайты), а по размеру и первым
    мегабайтам: этого хватает, чтобы поймать ПОДМЕНУ файла, ради которой
    подпись и заводится -- статистика с другого чекпоинта уже была
    источником тихих ошибок (закон 15)."""
    h = hashlib.sha256()
    st = os.stat(ckpt_path)
    h.update(str(st.st_size).encode())
    with open(ckpt_path, "rb") as f:
        h.update(f.read(1 << 20))
    with open(corpus_path, "rb") as f:
        h.update(f.read())
    h.update(type(tokenizer).__name__.encode())
    return h.hexdigest()[:16]


def collect(ckpt_path, tokenizer, corpus_path=CORPUS, seq_len=SEQ_LEN,
            budget=TOKEN_BUDGET, cache=True, verbose=True):
    """E[x^2] по входным каналам -> {ключ_веса: tensor[in_features]}."""
    from rwkv_quant.models import rwkv7_ref as ref_mod
    from rwkv_quant.models.rwkv7_ref import RWKV7Ref

    sig = _signature(ckpt_path, tokenizer, corpus_path)
    cache_path = os.path.join(CACHE_DIR, "act_%s.pt" % sig)
    if cache and os.path.exists(cache_path):
        if verbose:
            print("[act_stats] кеш %s" % cache_path)
        return torch.load(cache_path), sig

    encode = _encoder(tokenizer)
    text = open(corpus_path, encoding="utf-8").read()
    chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", text) if c.strip()]
    wins = _windows(chunks, encode, seq_len, budget)
    if not wins:
        raise RuntimeError(
            "калибровочный корпус не дал ни одного окна на %d токенов -- "
            "проверьте токенизатор" % seq_len)
    data = torch.tensor(wins, dtype=torch.int32)

    model = RWKV7Ref(ckpt_path, device="cpu", dtype=torch.bfloat16)
    vocab = int(getattr(model, "vocab_size", 0) or 0)
    top = int(data.max())
    if vocab and top >= vocab:
        # ГРОМКО, а не предупреждением: токенизатор от другой модели даёт
        # осмысленные на вид, но чужие id, и статистика молча становится
        # шумом. Ровно этот сценарий уже стоил дня (закон 15).
        raise ValueError(
            "токенизатор не от этого чекпоинта: максимальный id %d при "
            "vocab_size=%d" % (top, vocab))
    if verbose:
        print("[act_stats] %d окон x %d = %d токенов, %d чанков, vocab %d"
              % (len(wins), seq_len, len(wins) * seq_len, len(chunks), vocab))

    ref_mod.ACT_RECORDER = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(data.shape[0]):
            model.forward(data[i:i + 1, :-1])
    stats = {k: (ss / max(n, 1)) for k, (ss, n) in ref_mod.ACT_RECORDER.items()}
    ref_mod.ACT_RECORDER = None
    del model
    if verbose:
        print("[act_stats] снято %d тензоров за %.0f с" % (len(stats),
                                                           time.time() - t0))
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(stats, cache_path)
    return stats, sig
