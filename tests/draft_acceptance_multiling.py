"""
Согласие драфта g1d-0.1b с целью g1h-1.5b ПО ЯЗЫКАМ.

Вариант tests/dev_draft_acceptance.py: та же методика (teacher-forcing,
доля совпадений argmax = вероятность принять первый токен раунда), но на
мультиязычном корпусе и с разбивкой ru/en/sr. Мотив: спекулятивка платит
за отвергнутые токены, а сербский на 1.5B заметно слабее -- если драфт на
нём разойдётся с целью, средний выигрыш будет не тот, что на английском.

ГРАНИЦА МЕТОДА (как в оригинале): для второго и дальше токенов раунда
драфт в бою идёт по СВОЕМУ продолжению, а здесь видит истинный текст.
E[принято] = sum p^i -- оценка СВЕРХУ. Если сверху мало, дальше можно не
смотреть.

Модели грузятся по очереди: 1.5B bf16 ~3 ГБ, держать обе разом незачем.
Логиты считаются по одной последовательности (512 x 65536 x 4 = 134 МБ),
а не всем корпусом сразу (это было бы 5 ГБ).

    python tests/draft_acceptance_multiling.py
"""
import gc
import os
import subprocess
import sys

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-metal"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mlx.core as mx  # noqa: E402

TARGET = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
DRAFT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")


def mem(tag):
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True).stdout.strip()
    mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                        text=True).stdout
    free = [l for l in mp.splitlines() if "free percentage" in l]
    print(f"  [mem/{tag}] {sw} | {free[0].strip() if free else ''}", flush=True)


def argmax_per_seq(pth, data):
    """argmax логитов на каждой позиции, по последовательности за раз."""
    from rwkv_metal.model.convert import load_pretrained
    model, cfg = load_pretrained(pth, verbose=False)
    assert model is not None, f"не сконвертировалась: {pth}"
    print(f"  {os.path.basename(pth)}: L={cfg.n_layer} D={cfg.n_embd}", flush=True)
    mem("модель загружена")
    out = []
    for i in range(data.shape[0]):
        logits = model(mx.array(data[i:i + 1].astype(np.int64)))
        am = mx.argmax(logits[0], axis=-1)
        mx.eval(am)
        out.append(np.array(am.tolist(), dtype=np.int64))
        del logits, am
    del model
    gc.collect()
    mx.clear_cache()
    return np.stack(out)


def main():
    blob = torch.load(CORPUS)
    data, langs = blob["tokens"].numpy(), np.array(blob["lang"])
    print(f"корпус {data.shape}", flush=True)
    mem("старт")

    print("цель:", flush=True)
    a_tgt = argmax_per_seq(TARGET, data)
    print("драфт:", flush=True)
    a_drf = argmax_per_seq(DRAFT, data)
    mem("обе модели выгружены")

    match = (a_tgt == a_drf)
    print(f"\n{'язык':<6}{'позиций':>9}{'согласие p':>13}"
          + "".join(f"{'E[k='+str(k)+']':>11}" for k in (2, 4, 6, 8)))
    for lang in list(sorted(set(blob["lang"]))) + ["ALL"]:
        sel = slice(None) if lang == "ALL" else (langs == lang)
        m = match[sel]
        p = float(m.mean())
        row = "".join(f"{sum(p**i for i in range(1, k+1)) + 1:>11.2f}"
                      for k in (2, 4, 6, 8))
        print(f"{lang:<6}{m.size:>9}{p:>13.4f}{row}")
    print("\nE[k] -- ожидаемое число принятых токенов за раунд, ВКЛЮЧАЯ "
          "бонус-токен цели; оценка сверху.")


if __name__ == "__main__":
    main()
