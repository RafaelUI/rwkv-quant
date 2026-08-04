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
    print("ориентация: манифест против rwkv7_ref.py")
    suffixes = transposed_keys_from_ref()
    check("ref действительно что-то транспонирует", bool(suffixes),
          f"{sorted(suffixes)}")
    if manifest.get("naming") != "world":
        print("  (чекпоинт не world -- транспозиций быть не должно)")

    bad, n_true = [], 0
    for key, m in manifest["tensors"].items():
        parts = key.split(".")
        suffix = parts[-1] if len(parts) == 4 and parts[2] == "att" else None
        want = (manifest.get("naming") == "world" and suffix in suffixes)
        got = codec.is_transposed(manifest, key)
        n_true += bool(got)
        if got != want:
            bad.append(f"{key}: манифест {got}, ref {want}")
    check(f"флаги совпали с ref ({n_true} транспонированных)", not bad,
          "; ".join(bad[:3]))
    # у world-чекпоинта транспонированных обязано быть много: 8 матриц на
    # слой минус v1/v2 нулевого слоя. Ноль означал бы, что регулярка мимо
    if manifest.get("naming") == "world":
        check("транспонированные найдены", n_true > 0)


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
    tbad = [k for k in common
            if codec.is_transposed(m1, k) != codec.is_transposed(m2, k)]
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
