"""Согласие драфта g1d-0.1b с целью g1h-1.5b на РЕАЛЬНОМ тексте.

Один вопрос: сколько токенов драфта примет жадная спекулятивка. Ответ на
него — не скорость, а КАЧЕСТВО совпадения argmax, и меряется он без
единого кернеля, квантования и тайминга.

Метод — teacher-forcing: обе модели гонятся ПАРАЛЛЕЛЬНЫМ проходом по одному
и тому же реальному тексту, на каждой позиции берётся argmax. Доля
совпадений p — это ровно вероятность принять ПЕРВЫЙ токен раунда, потому
что там префикс у обеих моделей одинаков по построению.

ЧЕСТНАЯ ГРАНИЦА МЕТОДА. Для второго и дальше токенов раунда драфт в бою
идёт по СВОЕМУ продолжению, а здесь он видит истинный текст. Значит
E[принято] ниже считается по модели независимых испытаний с той же p и
является ОЦЕНКОЙ СВЕРХУ. Если сверху уже мало — дальше можно не смотреть;
если много — нужен настоящий цикл, и это уже другая работа.

Модели грузятся ПО ОЧЕРЕДИ: 1.5B bf16 это ~3 ГБ, плюс torch-free
zip-загрузчик даёт пик ~2.3x файла (закон из NEXT_SESSION rwkv-quant).
Держать обе разом на 16 ГБ незачем.

    python tests/dev_draft_acceptance.py [n_tokens]
"""
import sys, os, gc
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))
sys.path.insert(0, os.path.expanduser("~/Develop/WKV-kvant"))
import numpy as np
import mlx.core as mx

TARGET = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
DRAFT  = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
VOCAB  = os.path.expanduser("~/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
TXT    = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.txt")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512


def tokens():
    from world_tokenizer import RWKV_WORLD_TOKENIZER
    tok = RWKV_WORLD_TOKENIZER(VOCAB)
    text = open(TXT, encoding="utf-8").read()
    ids = tok.encode(text)
    print(f"корпус: {len(text)} символов -> {len(ids)} токенов, берём {N}")
    return ids[:N]


def argmax_per_position(pth, ids):
    """argmax логитов на каждой позиции одного параллельного прохода."""
    from rwkv_metal.model.convert import load_pretrained
    model, cfg = load_pretrained(pth, verbose=False)
    assert model is not None, f"не сконвертировалась: {pth}"
    print(f"  {os.path.basename(pth)}: L={cfg.n_layer} D={cfg.n_embd} "
          f"vocab={cfg.vocab_size}")
    idx = mx.array(np.array(ids, dtype=np.int64)[None, :])
    logits = model(idx)
    am = mx.argmax(logits[0], axis=-1)
    mx.eval(am)
    out = np.array(am.tolist(), dtype=np.int64)
    del logits, am, model
    gc.collect()
    return out


def main():
    ids = tokens()
    print("цель:")
    a_tgt = argmax_per_position(TARGET, ids)
    print("драфт:")
    a_drf = argmax_per_position(DRAFT, ids)

    n = min(len(a_tgt), len(a_drf))
    tgt, drf = a_tgt[:n], a_drf[:n]
    p = float((tgt == drf).mean())

    print(f"\nпозиций сравнено: {n}")
    print(f"СОГЛАСИЕ argmax драфт<->цель: {p:.4f}")

    nxt = np.array(ids[1:1 + n], dtype=np.int64)
    m = min(len(nxt), n)
    print(f"  (цель угадывает сам корпус в {float((tgt[:m] == nxt[:m]).mean()):.4f}, "
          f"драфт — в {float((drf[:m] == nxt[:m]).mean()):.4f})")

    print("\nОЦЕНКА СВЕРХУ на принятые токены за раунд:")
    print("   k   E[принято]   с бонусом")
    for k in (2, 4, 6, 8):
        e = sum(p ** i for i in range(1, k + 1))
        print(f"  {k:2d}      {e:5.2f}       {e + 1:5.2f}")

    print("\nГрубый перевод в мс/ток (раунд = верифай T=k+1, полка 7.3-7.9):")
    print("  сегодня без спекулятивки: 14.15")
    for k in (4, 8):
        e = sum(p ** i for i in range(1, k + 1)) + 1
        for shelf in (7.3, 7.9):
            print(f"  k={k}, полка {shelf}: {shelf * (k + 1) / e:6.2f} мс/ток")


if __name__ == "__main__":
    main()
