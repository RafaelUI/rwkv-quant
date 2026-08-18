"""Сборка варианта REDUCTION с `w/a/v_lora` на ВОСЬМИ битах вместо шести.

ПОЧЕМУ ЭТО БЕСПЛАТНО ПО БАЙТАМ. Ветки лежат в контейнере `asym gw64`: коды
там uint8 (суб-байтовая упаковка только при bits<=4), scale/min -- fp32 на
блок 64, поэтому контейнер стоит РОВНО 9.000 бит/вес при пяти, шести и
восьми битах одинаково (`probe_schema_cost`, 12.08; проверено побайтово на
полной модели 1.5B: 1 255 797 192 байта при шести и при восьми, разные
sha256). Битность там -- чистый параметр качества, и сейчас он потрачен
впустую.

ГЕЙТ ВСТРОЕН В САМУ СБОРКУ: файл ОБЯЗАН выйти того же размера, что
эталонный, байт в байт. Если размер разошёлся -- либо арифметика
контейнера не та, что записана, либо пресет подменился (закон 30), и в
обоих случаях замер на этом файле смысла не имеет.

ЛОВУШКА, ИЗ-ЗА КОТОРОЙ ЭТОТ СКРИПТ ПРОВЕРЯЕТ act_stats ПЕРВЫМ ДЕЛОМ:
`presets.REDUCTION` ссылается на `/tmp/act_stats_1p5b.pt`, а /tmp ребут не
переживает. Без файла AW-режимы молча вырождаются в обычный поиск --
предупреждение будет, сборка пройдёт, и получится ДРУГОЙ пресет под тем же
именем (закон 15).

    python make_lora8.py [out.rwkvq]
"""
import os
import sys
import copy

sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-quant"))

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_lora8.rwkvq"
CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
REF_BYTES = 1435062048          # /tmp/reduction_new.rwkvq, пресет 16.08


def main():
    import rwkv_quant
    from rwkv_quant import presets

    base = presets.REDUCTION
    ap = base.act_stats_path
    assert ap and os.path.exists(os.path.expanduser(ap)), (
        "НЕТ %s -- AW выродится в обычный поиск, и это будет ДРУГОЙ пресет "
        "под тем же именем (закон 15). Сначала tests/restore_tmp.sh" % ap)
    print("act_stats на месте: %s" % ap)

    # QuantConfig -- НЕ dataclass, `replace` на нём падает. Копия с
    # точечной правкой словаря `bits` надёжнее конструирования заново:
    # конструктор прогнал бы clip_percentiles/group_scale через
    # expand_aliases повторно, а копия гарантирует, что всё остальное
    # осталось тем же объектом, а не «похожим».
    cfg = copy.deepcopy(base)
    for g in ("w_lora", "a_lora", "v_lora"):
        cfg.bits[g] = 8
    diff = {k: (base.bits[k], cfg.bits[k]) for k in base.bits
            if base.bits[k] != cfg.bits[k]}
    assert set(diff) == {"w_lora", "a_lora", "v_lora"}, (
        "изменилось не то, что заказано: %s" % diff)
    print("битность изменена ровно у %s, g_lora остаётся %d"
          % (sorted(diff), cfg.bits["g_lora"]), flush=True)

    rwkv_quant.quantize(CKPT, OUT, config=cfg, real_gw=True, verbose=True)

    got = os.path.getsize(OUT)
    print("\n%s: %d байт" % (OUT, got))
    print("эталон  : %d байт" % REF_BYTES)
    if got == REF_BYTES:
        print("ГЕЙТ ЗЕЛЁНЫЙ: размер тот же байт в байт -- битность веток в "
              "контейнере asym gw64 действительно бесплатна")
    else:
        print("ГЕЙТ КРАСНЫЙ: расхождение %+d байт (%.3f%%). Записанное "
              "«9.000 бит/вес при 5, 6 и 8 одинаково» на этом файле НЕ "
              "воспроизвелось -- разбираться, а не мерить дальше."
              % (got - REF_BYTES, 100.0 * (got - REF_BYTES) / REF_BYTES))
        sys.exit(1)


if __name__ == "__main__":
    main()
