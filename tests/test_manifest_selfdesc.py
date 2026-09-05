"""
Гейт шага 6: манифест описывает сам себя, и описывает ПРАВДУ.

Три утверждения, каждое проверяется против независимого источника, а не
против того же кода, что поле заполняет.

  1. `transposed` -- сверяется с ИСХОДНИКОМ models/rwkv7_ref.py: оттуда
     регуляркой достаётся множество ключей, к которым ref реально
     применяет .T при загрузке world-чекпоинта. Если кто-то поменяет
     ориентацию в ref и забудет про формат, гейт покраснеет. Сверять
     writer с codec.is_raw_lora_world было бы тавтологией: одна таблица
     имён проверяет сама себя.
  2. `n_blocks` -- сверяется с фактическими формами буферов и отдельно
     с ceil(IN/gs). Гейт ТРЕБУЕТ, чтобы в чекпоинте нашёлся хотя бы один
     тензор с n_blocks != IN // gs: иначе проверка ничего не значит, а
     именно этот случай (`att.w1` [2048, 96] при gs=64) и ломает
     наивного читателя.
  3. `config` -- роундтрип структурой, плюс демонстрация того, ЗАЧЕМ он:
     показывается, что config_repr тех же полей не содержит.

Плюс совместимость: манифест v1 (без новых полей) должен читаться, а
codec -- выводить недостающее и давать тот же деквант. Файл v1 для этого
делается из v2 правкой ОДНОГО заголовка (downgrade_to_v1), буферы
побайтово те же -- иначе сравнивались бы два разных чекпоинта.

    python tests/test_manifest_selfdesc.py <model.rwkvq>
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from rwkv_quant.formats import codec  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402
from rwkv_quant.formats.writer import config_to_json, save_rwkvq  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def transposed_keys_from_ref():
    """Суффиксы world-ключей, к которым rwkv7_ref применяет .T. Читаем
    исходник: это единственный независимый источник истины про ориентацию."""
    path = os.path.join(os.path.dirname(__file__), "..",
                        "rwkv_quant", "models", "rwkv7_ref.py")
    src = open(path).read()
    # интересует только ветка world (get(ap + "...")), custom берёт
    # .weight-ключи уже в ориентации nn.Linear
    return set(re.findall(r'get\(ap \+ "(\w+)"\)\.T', src))


def test_transposed(manifest):
    print("ориентация: флаг манифеста против ФОРМ, записанных в нём же")
    # Прежняя редакция сверяла флаг с таблицей имён из rwkv7_ref.py. Эта
    # таблица описывает ЧЕКПОИНТ (.pth), а не контейнер, и держалась только
    # пока writer квантовал LoRA в сырой раскладке. С 28.08 LoRA может
    # лежать уже транспонированной, и тогда прежняя сверка краснела на
    # исправном файле, а на порче -- зеленела. Сверяем флаг с данными.
    ne = int(manifest["n_embd"])
    lora = ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")
    bad, seen, n_true = [], 0, 0
    for key, m in manifest["tensors"].items():
        parts = key.split(".")
        if not (len(parts) == 4 and parts[2] == "att" and parts[3] in lora):
            continue
        seen += 1
        shape = tuple(int(x) for x in m["shape"])
        got = codec.is_transposed(manifest, key)
        n_true += bool(got)
        res = shape[::-1] if got else shape
        # после разрешения флага потребитель обязан получить [out, in]:
        # у A-матриц (суффикс 1) редукция идёт по n_embd, значит n_embd --
        # последняя ось; у B-матриц (суффикс 2) n_embd -- ось выхода.
        ok = len(res) == 2 and (res[-1] == ne if parts[3].endswith("1")
                                else res[0] == ne)
        if not ok:
            bad.append("%s: хранится %s, флаг %s, после разрешения %s, "
                       "n_embd=%d" % (key, shape, got, res, ne))
    check("LoRA-матрицы найдены", seen > 0, "их %d" % seen)
    check("флаг согласован с формами (%d транспонированных из %d)"
          % (n_true, seen), not bad, "; ".join(bad[:3]))


def test_sidecar_transposed(ckpt, manifest):
    """Сайдкар MLX ДОНОСИТ ориентацию, а не роняет её."""
    # До 05.09 export_mlx поля transposed не писал вовсе: манифест .rwkvq
    # ориентацию знает, а потребитель сайдкара мог узнать её только из
    # зашитой у себя таблицы имён. Ровно этим болел rwkv-metal до 1bf09d6
    # и болеет RwkvqFullConvert.swift:79-93.
    from rwkv_quant.formats.export_mlx import _export_one
    keys = [k for k in manifest["tensors"]
            if k.endswith(("att.w1", "att.a1", "att.key.weight",
                           "ffn.key.weight"))][:8]
    check("ключи для проверки сайдкара нашлись", len(keys) > 0, str(len(keys)))
    bad = []
    for k in keys:
        meta = _export_one(k, ckpt.tensors[k], {})
        if "transposed" not in meta:
            bad.append(k + ": поля нет")
        elif bool(meta["transposed"]) != codec.is_transposed(manifest, k):
            bad.append(k)
    check("сайдкар повторяет ориентацию источника", not bad, "; ".join(bad[:3]))
    if not keys:
        return

    # МУТАЦИЯ. Без неё гейт был бы зелёным и на захардкоженном False:
    # в файлах новой раскладки ВСЕ ключи не транспонированы, так что
    # совпадение значений само по себе ничего не доказывает.
    k = keys[0]
    qt = ckpt.tensors[k]
    was = qt.transposed
    try:
        qt.transposed = not bool(was)
        got = _export_one(k, qt, {}).get("transposed")
    finally:
        qt.transposed = was
    check("флаг следует за источником (мутация)",
          got is not None and bool(got) == (not bool(was)),
          "%s: было %s, после переворота %s" % (k, was, got))

    # Неизвестная ориентация обязана падать ГРОМКО: сайдкар, выгруженный
    # без неё, у потребителя ошибётся тихо (закон 15).
    try:
        qt.transposed = None
        _export_one(k, qt, {})
        loud = False
    except ValueError:
        loud = True
    finally:
        qt.transposed = was
    check("неизвестная ориентация падает громко", loud)


def test_n_blocks(manifest, arrays):
    print("\nn_blocks: манифест против форм буферов")
    bad, interesting = [], []
    for key, m in manifest["tensors"].items():
        kind = m["kind"]
        if kind not in ("sb6", "asym"):
            continue
        nb, gs, IN = m["n_blocks"], m["gw_gs"], m["shape"][1]
        real = (arrays[f"{key}::gw_qsqm"].shape[-2] * m["gw_sb"]
                if kind == "sb6" else arrays[f"{key}::gw_scale"].shape[-1])
        if nb != real:
            bad.append(f"{key}: {nb} против {real} в буфере")
        elif nb != -(-IN // gs):
            bad.append(f"{key}: {nb} != ceil({IN}/{gs})")
        if nb != IN // gs:
            interesting.append(f"{key} [{m['shape']}] gs={gs}: {nb} против {IN // gs}")
    check("n_blocks совпал с буферами", not bad, "; ".join(bad[:3]))
    check("нашёлся тензор, где деление нацело врёт", bool(interesting),
          interesting[0] if interesting else
          "все размерности кратны gs -- гейт вырожден, взять чекпоинт с LoRA")
    if interesting:
        print(f"       таких тензоров {len(interesting)}")


def test_config(manifest, ckpt):
    print("\nconfig: структура против лосси-repr")
    cfg = manifest.get("config")
    if cfg is None:
        check("config в манифесте", False, "None -- файл v1?")
        return
    check("config: биты по группам", bool(cfg["bits"]), str(cfg["bits"]))

    # то, ради чего всё: эти поля в repr() не попадают вовсе
    lost = [k for k in ("group_scale", "group_scale_mode", "act_stats_path")
            if cfg.get(k) and str(cfg[k]) not in manifest.get("config_repr", "")]
    check("repr() терял поля, структура их сохранила", bool(lost),
          ", ".join(lost) if lost else "конфиг пуст -- нечего терять")

    back = ckpt.config
    check("config собрался обратно в QuantConfig", back is not None)
    if back is not None:
        check("роундтрип config без потерь", config_to_json(back) == cfg)


def downgrade_to_v1(src_v2, dst_v1):
    """Тот же файл с манифестом v1: правится ТОЛЬКО заголовок, буферы
    побайтово те же. Так совместимость проверяется на одном и том же
    чекпоинте -- сравнивать два разных пресета было бы бессмысленно."""
    import json
    import struct
    with open(src_v2, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        rest = f.read()
    m = json.loads(header["__metadata__"]["rwkvq"])
    m["format_version"] = 1
    for k in ("config", "tokenizer"):
        m.pop(k, None)
    for t in m["tensors"].values():
        for k in ("n_blocks", "transposed"):
            t.pop(k, None)
    header["__metadata__"]["rwkvq"] = json.dumps(m)
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)          # safetensors требует выравнивания
    with open(dst_v1, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + rest)


def test_v1_compat(path_v1, ref_manifest):
    print(f"\nсовместимость с манифестом v1: {path_v1}")
    m1, a1 = codec.open_rwkvq(path_v1)
    ver = m1.get("format_version")
    check("это действительно v1", ver == 1, f"v{ver}")
    check("в v1 полей и нет", "n_blocks" not in next(iter(m1["tensors"].values())))

    # codec обязан вывести недостающее и дать ТОТ ЖЕ ответ, что на v2
    m2, a2 = ref_manifest
    common = sorted(set(m1["tensors"]) & set(m2["tensors"]))
    bad, n = [], 0
    for key in common:
        w1 = codec.dequant_key(m1, a1, key)
        w2 = codec.dequant_key(m2, a2, key)
        n += 1
        if w1.shape != w2.shape or not bool((w1 == w2).all()):
            bad.append(key)
    check(f"деквант v1 == деквант v2 ({n} тензоров)", not bad,
          ", ".join(bad[:3]))
    raw_layout = all(codec.is_transposed(m2, k) for k in common
                     if codec.is_raw_lora_world(k))
    tbad = [] if not raw_layout else [k for k in common
            if codec.is_transposed(m1, k) != codec.is_transposed(m2, k)]
    if not raw_layout:
        print("  (файл в новой раскладке LoRA -- v1 её выразить не может, "
              "сверка выводимости неприменима)")
    check("transposed выводится для v1 так же, как записан в v2", not tbad,
          ", ".join(tbad[:3]))


def main():
    src = sys.argv[1]
    out = src + ".selfdesc"
    ckpt = load_raw(src)

    # у файлов v1 конфига нет вовсе (был только лосси-repr), так что для
    # проверки плумбинга берём настоящий пресет -- он как раз содержит и
    # group_scale, и act_stats_path, то есть ровно то, что repr() терял
    cfg = ckpt.config
    if cfg is None:
        from rwkv_quant.presets import PRESETS
        cfg = PRESETS["reduction"]
        print(f"(в {src} конфига нет -- пишем пресет reduction)")
    save_rwkvq(ckpt, out, config=cfg, tokenizer="rwkv_vocab_v20230424.txt")

    manifest, arrays = codec.open_rwkvq(out)
    print(f"{out}: манифест v{manifest['format_version']}, "
          f"{len(manifest['tensors'])} тензоров, naming={manifest['naming']}, "
          f"токенайзер {manifest.get('tokenizer')!r}\n")

    test_transposed(manifest)
    test_sidecar_transposed(ckpt, manifest)
    test_n_blocks(manifest, arrays)
    test_config(manifest, load_raw(out))

    v1 = out + ".v1"
    downgrade_to_v1(out, v1)
    test_v1_compat(v1, (manifest, arrays))

    os.remove(v1)
    os.remove(out)
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else f"ПРОВАЛЕН: {', '.join(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
