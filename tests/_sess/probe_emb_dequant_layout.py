"""ЦЕНА ПАМЯТИ dense-ДЕКВАНТА ОДНОГО ТЕНЗОРА, ПО РАСКЛАДКАМ (сессионный).

ЗАЧЕМ. Перемер шага QLoRA на 2.9B (27.08) дал footprint 8390 МБ против
9487 у прежней базы -- при том, что новый файл на 253 МБ БОЛЬШЕ и держит
в MLX на 250 МБ больше кодов. Значит на загрузке исчез транзиент порядка
гигабайта. Единственный КРУПНЫЙ тензор, у которого раскладка различается
между файлами, -- emb.weight (sb6@6 в прежнем, sym@8 в нынешнем); head в
обоих sym@8. Здесь это проверяется прямо, а не выводится: деквантуется
РОВНО ОДИН тензор, пик снимается снаружи `/usr/bin/time -l` (закон 22).

КОНТРОЛЬ ВКЛЮЧЕНИЯ ОБЯЗАТЕЛЕН (закон 38 наоборот: «ничего не посчиталось»
и «посчиталось дёшево» выглядят одинаково). Печатается раскладка, форма,
битность и отпечаток результата -- сумма модулей ПО СРЕЗУ и в fp32;
по всей fp16-таблице это дало бы inf, а inf == inf молча выродил бы
контроль в «плечи совпали» (грабли 26.08).

    /usr/bin/time -l python probe_emb_dequant_layout.py <файл.rwkvq> [ключ] [banded|whole]
"""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-quant"))

import torch  # noqa: E402

from rwkv_quant.formats import reader  # noqa: E402

PATH = sys.argv[1]
KEY = sys.argv[2] if len(sys.argv) > 2 else "emb.weight"
MODE = sys.argv[3] if len(sys.argv) > 3 else "banded"


def main():
    ck = reader.load_raw(PATH)
    qt = ck.tensors[KEY]
    mode = getattr(qt, "gw_mode", "") or "rtn"
    print(f"файл {os.path.basename(os.path.realpath(PATH))}")
    print(f"  {KEY}: раскладка {mode}, бит {qt.bits}, форма {tuple(qt.shape)}, "
          f"режим декванта {MODE}")

    t0 = time.time()
    if MODE == "whole":
        w = reader._dequantize_one(qt).to(torch.float16)
    else:
        w = reader.dequantize_banded(qt, dtype=torch.float16)
    dt = time.time() - t0

    # отпечаток по СРЕЗУ и в fp32 -- по всей fp16-таблице это inf
    fp = w[:64].float().abs().sum().item()
    print(f"  деквант {dt*1e3:.0f} мс, результат {w.dtype}, "
          f"{w.numel()*w.element_size()/1e6:.1f} МБ, отпечаток {fp:.6f}")
    if not (fp > 0.0) or fp != fp:
        raise SystemExit("ОТПЕЧАТОК ВЫРОЖДЕН -- контроль включения не прошёл")


if __name__ == "__main__":
    main()
