# -*- coding: utf-8 -*-
"""AW НА `emb` -- НЕ ОПЕРАЦИЯ, И ГЕЙТ СЛЕДИТ, ЧТОБЫ ЭТО НЕ ИЗМЕНИЛОСЬ МОЛЧА.

Пресет REDUCTION до 26.08 объявлял `emb: "sym_aw"`. Статистики активаций
для `emb.weight` не существует ни в одном файле и не может существовать:
у таблицы эмбеддингов вход -- индексы, E[x^2] по ним не определено, а
`ACT_RECORDER` в `rwkv7_ref` пишет только по входу линейных слоёв.
`groupwise.get_ex2` при ОТСУТСТВУЮЩЕМ КЛЮЧЕ возвращает None ТИХО (закон 15
покрывал лишь несовпадение длины), поэтому `sym_aw` на emb всё это время
исполнялся как `sym`. Надпись снята.

ЗАЧЕМ ГЕЙТ ПОСЛЕ ТОГО, КАК НАДПИСЬ СНЯТА. Затем, что обратная правка
дешева и незаметна: стоит кому-нибудь дописать `emb.weight` в файл
статистики -- и `emb` начнёт квантоваться ИНАЧЕ, а все числа, записанные
под именем пресета, перестанут воспроизводиться (закон 30). Гейт краснеет
ровно в этот момент.

УТВЕРЖДЕНИЕ: на РЕАЛЬНОМ `emb.weight` режимы `sym` и `sym_aw` дают
побайтово один результат.
КОНТРОЛЬ ОСМЫСЛЕННОСТИ: на `head.weight`, где статистика ЕСТЬ, те же два
режима обязаны РАЗОЙТИСЬ -- иначе инструмент не отличает режимы вовсе и
первое утверждение зелено само по себе (закон 8).

    python tests/test_emb_aw_is_noop.py [checkpoint.pth] [act_stats.pt]
"""
import copy
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rwkv_quant.presets import REDUCTION  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor, TENSOR_FIELDS  # noqa: E402

CKPT = (sys.argv[1] if len(sys.argv) > 1 else
        os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
STATS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/act_stats_1p5b_ml.pt"


def digest(qt):
    h = hashlib.md5()
    for f in TENSOR_FIELDS:
        v = getattr(qt, f, None)
        h.update(b"\x00" if v is None else v.contiguous().numpy().tobytes())
    return h.hexdigest()


def cfg_with(group, mode):
    c = copy.deepcopy(REDUCTION)
    c.act_stats_path = STATS
    c.group_scale_mode[group] = mode
    return c


def main():
    if not os.path.exists(STATS):
        raise SystemExit("нет %s -- запустите tests/restore_tmp.sh" % STATS)
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    bad = []
    for key, group, must_differ in (("emb.weight", "emb", False),
                                    ("head.weight", "head", True)):
        w = sd[key]
        a = digest(quantize_tensor(key, w, cfg_with(group, "sym_aw"), real_gw=True))
        b = digest(quantize_tensor(key, w, cfg_with(group, "sym"), real_gw=True))
        same = (a == b)
        ok = (same != must_differ)
        print("  %-12s sym_aw=%s sym=%s -> %s  %s"
              % (key, a[:12], b[:12],
                 "одинаково" if same else "различно",
                 "ок" if ok else "ПРОВАЛ"))
        if not ok:
            bad.append(key)
    if bad:
        if "emb.weight" in bad:
            print("\n[FAIL] на emb.weight появилась статистика активаций: "
                  "`emb` теперь квантуется иначе, чем во всех записанных "
                  "замерах. Либо уберите её, либо перемерьте пресет "
                  "целиком и перепишите числа (закон 30).")
        if "head.weight" in bad:
            print("\n[FAIL] на head.weight режимы НЕ разошлись: инструмент "
                  "не отличает sym от sym_aw, и утверждение про emb ничего "
                  "не значит (закон 8).")
        return 1
    print("\n[OK] AW на emb -- не операция; на head режимы различимы")
    return 0


if __name__ == "__main__":
    sys.exit(main())
