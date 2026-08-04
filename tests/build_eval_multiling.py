"""
Мультиязычный eval-корпус из ~/Develop/test.txt (ru / en / sr-cyrillic).

Зачем отдельно от build_eval_corpus_world.py (WKV-kvant): тот корпус --
24x128 токенов чистого английского из harrier-пар. Для приложения важно
другое: world-vocab кодирует сербскую кириллицу заметно хуже русской
(редкие токены, байтовые фоллбэки), а деградация от квантования emb/head
бьёт именно по редким токенам. Агрегированный ppl это прячет, поэтому
здесь язык каждой последовательности сохраняется рядом с токенами и ppl
считается раздельно.

Раскладка: каждый чанк токенизируется целиком и режется окнами по
SEQ_LEN БЕЗ пересечения границ чанков -- иначе на стыке двух текстов
модель платит ppl-штраф, не имеющий отношения к квантованию.

Разбиение calib/eval -- ПО ЧАНКАМ, не по окнам: AW-режимы обоих пресетов
подбирают scale под статистику активаций, и если она снята с того же
документа, на котором потом меряется ppl, выигрыш AW окажется завышенным.
В калибровку уходят самые короткие чанки каждого языка, пока не наберётся
CALIB_TOKENS (большие остаются мерить).

Выход:
  eval_corpus_multiling.pt = {"tokens": int32 [N, SEQ_LEN], "lang": [N],
                              "seq_len": SEQ_LEN}   -- измерительная часть
  act_calib_multiling.pt   = int32 [M, SEQ_LEN]     -- голый тензор для
                              tests/collect_act_stats.py (он делает
                              torch.load(...)[a:b] и ждёт тензор)
"""
import os
import re
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
from rwkv_metal.tokenizer.world_tokenizer import WorldTokenizer  # noqa: E402

SRC = os.path.expanduser("~/Develop/test.txt")
OUT = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
OUT_CALIB = os.path.expanduser("~/Develop/WKV-kvant/act_calib_multiling.pt")
SEQ_LEN = 512
CALIB_TOKENS = 2000   # на язык; E[x^2] по каналам сходится быстро

# Сербский здесь в кириллице, поэтому от русского его отличает не алфавит,
# а специфические буквы ј/љ/њ/ћ/џ/ђ, которых в русском нет вовсе.
_SR = re.compile(r"[јљњћџђЈЉЊЋЏЂ]")
_CYR = re.compile(r"[Ѐ-ӿ]")


def detect_lang(text: str) -> str:
    cyr = len(_CYR.findall(text))
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    if cyr <= lat:
        return "en"
    # доля сербо-специфичных букв среди кириллицы: в русском ровно 0
    return "sr" if len(_SR.findall(text)) / max(cyr, 1) > 0.005 else "ru"


def main():
    text = open(SRC, encoding="utf-8").read()
    chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", text) if c.strip()]
    print(f"чанков в {SRC}: {len(chunks)}")

    tok = WorldTokenizer()
    encoded = [(i, detect_lang(ch), tok.encode(ch)) for i, ch in enumerate(chunks)]

    # калибровочные чанки: самые короткие каждого языка, пока не наберём
    # CALIB_TOKENS -- длинные остаются измерительной части
    calib_ids = set()
    for lang in sorted({l for _, l, _ in encoded}):
        got = 0
        for i, l, ids in sorted(encoded, key=lambda e: len(e[2])):
            if l != lang or got >= CALIB_TOKENS or len(ids) < SEQ_LEN:
                continue
            calib_ids.add(i)
            got += len(ids)

    def windows(ids):
        """Окна SEQ_LEN внутри одного чанка + добор хвоста перекрытием."""
        out = [ids[w * SEQ_LEN:(w + 1) * SEQ_LEN] for w in range(len(ids) // SEQ_LEN)]
        tail = len(ids) % SEQ_LEN
        if tail >= SEQ_LEN // 2 and len(ids) >= SEQ_LEN:
            out.append(ids[-SEQ_LEN:])
        return out

    seqs, langs, chunk_ids, calib_seqs = [], [], [], []
    per_lang_tokens = {}
    for i, lang, ids in encoded:
        per_lang_tokens[lang] = per_lang_tokens.get(lang, 0) + len(ids)
        wins = windows(ids)
        role = "CALIB" if i in calib_ids else "eval"
        if i in calib_ids:
            calib_seqs.extend(wins)
        else:
            seqs.extend(wins)
            langs.extend([lang] * len(wins))
            # id чанка обязателен: агрегат по языку на 9 окнах из ДВУХ
            # документов неотличим от эффекта одного документа. Разбивка
            # по чанкам показывает, разъезжаются ли тексты одного языка.
            chunk_ids.extend([i] * len(wins))
        print(f"  чанк {i:2d} lang={lang} {role} chars={len(chunks[i]):6d} "
              f"tokens={len(ids):6d} ({len(chunks[i])/max(len(ids),1):.2f} chars/tok)"
              f" -> {len(wins)} окон")
    tokens = torch.tensor(seqs, dtype=torch.int32)
    preview = {i: chunks[i][:70].replace("\n", " ") for i in set(chunk_ids)}
    torch.save({"tokens": tokens, "lang": langs, "chunk": chunk_ids,
                "preview": preview, "seq_len": SEQ_LEN}, OUT)
    calib = torch.tensor(calib_seqs, dtype=torch.int32)
    torch.save(calib, OUT_CALIB)
    print(f"\ncalib: {calib.shape[0]} посл. -> {OUT_CALIB}")

    print(f"\nитого {tokens.shape[0]} последовательностей x {SEQ_LEN} = "
          f"{tokens.numel()} токенов -> {OUT}")
    for lang in sorted(set(langs)):
        n = sum(1 for x in langs if x == lang)
        print(f"  {lang}: {n:3d} посл. ({n * SEQ_LEN:6d} токенов в корпусе, "
              f"{per_lang_tokens[lang]} до нарезки)")


if __name__ == "__main__":
    main()
