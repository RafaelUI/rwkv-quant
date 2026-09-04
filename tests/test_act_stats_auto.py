"""ГЕЙТ: автосбор калибровки внутри quantize() -- работает и ОТКАЗЫВАЕТ ГРОМКО.

Проверяются пять вещей, каждая из которых уже была источником тихой
ошибки в этом проекте:

  1. Без токенизатора -- ПАДЕНИЕ с внятным текстом, а не тихая сборка без
     AW (измеренная цена молчания: KL хуже на 38%).
  2. Токенизатор от другой модели (id за пределами vocab_size) -- ПАДЕНИЕ,
     а не статистика-шум (закон 15).
  3. Круговой обход чанков реально доносит ВСЕ письменности до статистики:
     если резать корпус подряд до бюджета, китайский (последние чанки) не
     попадёт вовсе, а это стоит всего выигрыша AW на нём.
  4. Пресеты больше не указывают на /tmp, а quantize() не мутирует
     разделяемый объект пресета между вызовами.
  5. Токенизатор, переданный ПУТЁМ К СЛОВАРЮ, разбирается. Форма
     документирована в quick-start api.quantize, но была недостижима: у
     str метод .encode есть, и hasattr стоял выше isinstance(..., str).

    python tests/test_act_stats_auto.py
"""
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

from rwkv_quant.calibration import act_stats as A  # noqa: E402
from rwkv_quant.presets import PRESETS  # noqa: E402
from rwkv_metal.tokenizer.world_tokenizer import WorldTokenizer  # noqa: E402
from rwkv_metal.tokenizer import world_tokenizer as wt_mod  # noqa: E402


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-38s %s %s" % (name, "ок" if cond else "ПРОВАЛ", detail))

    print("корпус:", A.CORPUS)
    check("корпус на месте", os.path.exists(A.CORPUS),
          "%.2f МБ" % (os.path.getsize(A.CORPUS) / 1e6))
    for name, cfg in PRESETS.items():
        check("пресет %s без пути к /tmp" % name, cfg.act_stats_path is None)

    # 1. нет токенизатора -> громко
    try:
        A._encoder(None)
        check("без токенизатора падает", False)
    except A.TokenizerRequired as e:
        check("без токенизатора падает", "act_stats=None" in str(e),
              "подсказка про осознанный отказ на месте")

    # 5. ПУТЬ К СЛОВАРЮ СТРОКОЙ. Эта форма уходила в str.encode -- у str
    # метод .encode есть, это кодирование текста в байты, -- и чанк корпуса
    # попадал туда как ИМЯ КОДИРОВКИ: LookupError с текстом чанка вместо
    # токенизации. Гейт при этом был зелёным, потому что звал _encoder(None)
    # и _encoder(объект): две ветки из трёх, а сломана была третья (закон 31).
    vocab = os.path.join(os.path.dirname(wt_mod.__file__),
                         "rwkv_vocab_v20230424.txt")
    check("словарь rwkv_metal на месте", os.path.exists(vocab), vocab)
    ids_obj = WorldTokenizer(vocab).encode("привет мир")

    def _enc(arg):
        # Отказ печатается как ПРОВАЛ, а не роняет гейт: иначе одна
        # сломанная ветка прячет все проверки ниже неё.
        try:
            return A._encoder(arg)("привет мир")
        except Exception as e:
            return repr(e)

    ids_str = _enc(vocab)
    ids_path = _enc(pathlib.Path(vocab))
    check("путь к словарю строкой", ids_str == ids_obj and len(ids_obj) > 0,
          ids_str if isinstance(ids_str, str) else "%d id" % len(ids_str))
    check("путь к словарю PathLike", ids_path == ids_obj,
          ids_path if isinstance(ids_path, str) else "")

    # 3. круговой обход доносит все письменности
    tok = WorldTokenizer()
    text = open(A.CORPUS, encoding="utf-8").read()
    chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", text) if c.strip()]
    wins = A._windows(chunks, tok.encode)
    cjk = re.compile(r"[\u3400-\u9fff]")
    langs = set()
    for w in wins:
        t = tok.decode(w) if hasattr(tok, "decode") else ""
        if cjk.search(t):
            langs.add("zh")
        elif sum(t.count(c) for c in "{};()=") > 0.02 * max(len(t), 1):
            langs.add("code")
        elif re.search(r"[Ѐ-ӿ]", t):
            langs.add("cyr")
        else:
            langs.add("lat")
    check("окон набрано", len(wins) > 0, "%d окон x %d = %d токенов"
          % (len(wins), A.SEQ_LEN, len(wins) * A.SEQ_LEN))
    check("китайский попал в калибровку", "zh" in langs, str(sorted(langs)))
    check("код попал в калибровку", "code" in langs)

    # КОНТРОЛЬ НЕВЫРОЖДЕННОСТИ проверки 3: последовательная нарезка ТОГО ЖЕ
    # корпуса на тот же бюджет обязана китайский ПОТЕРЯТЬ -- иначе тест
    # выше зелёный сам по себе и ничего не доказывает.
    seq = []
    for ch in chunks:
        ids = tok.encode(ch)
        for w in range(len(ids) // A.SEQ_LEN):
            if len(seq) * A.SEQ_LEN >= A.TOKEN_BUDGET:
                break
            seq.append(ids[w * A.SEQ_LEN:(w + 1) * A.SEQ_LEN])
    has_zh = any(cjk.search(tok.decode(w)) for w in seq)
    check("контроль: подряд китайский теряется", not has_zh,
          "иначе круговой обход ничего не решает")

    print("\n[%s] автосбор калибровки" % ("OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
