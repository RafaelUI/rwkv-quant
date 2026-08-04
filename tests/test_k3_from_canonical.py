"""
Гейт: K3-интерлив, построенный из ДИСКОВОЙ раскладки на numpy, совпадает
с тем, что кладёт в сайдкар export_mlx через torch.

Зачем это решает. Пока K3 умел строить только export_mlx, потребители
формата (rwkv-metal, SwiftRWKV) были обязаны возить рядом с .rwkvq
отдельный сайдкар: сам .rwkvq они прочитать могли, а разложить его так,
как хочет ядро, -- нет. Если numpy-версия сходится бит-в-бит, сайдкар
перестаёт быть нужен кому-либо и остаётся лишь необязательным кешем.

Сверка содержательна, а не тавтологична: слева numpy из канонической
раскладки, справа torch через GwQuantLinear из того же .rwkvq. Два
независимых пути к одним байтам.

    python tests/test_k3_from_canonical.py <model.rwkvq> <сайдкар_без_расширения>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from rwkv_quant.formats import codec  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def main():
    rwkvq, sidecar = sys.argv[1], sys.argv[2]
    manifest, arrays = codec.open_rwkvq(rwkvq)
    ref = mx.load(sidecar + ".safetensors")
    ref_manifest = json.load(open(sidecar + ".json"))

    sb6 = [(k, m) for k, m in manifest["tensors"].items() if m["kind"] == "sb6"]
    common = [(k, m) for k, m in sb6 if f"{k}::qblk" in ref]
    print(f"{os.path.basename(rwkvq)}: sb6 {len(sb6)}, "
          f"есть в сайдкаре {len(common)}")
    if not common:
        print("пересечения нет -- сайдкар от другой модели?")
        return 1

    # по два представителя на форму
    seen, picked = {}, []
    for k, m in sorted(common):
        sh = tuple(m["shape"])
        if len(seen.setdefault(sh, [])) < 2:
            seen[sh].append(k)
            picked.append((k, m))

    n_bytes = 0
    for key, m in picked:
        OUT, IN = m["shape"]

        def buf(f):
            return arrays.get(f"{key}::{f}")

        qblk, qsqm, ddm, xbits = codec.sb6_to_k3(
            buf("codes_packed"), buf("gw_qsqm"), buf("gw_d"), buf("gw_dm"),
            shape=(OUT, IN), gs=m["gw_gs"], sb=m["gw_sb"],
            nb=m.get("n_blocks"), qh=buf("gw_qh"), qh2=buf("gw_qh2"))

        got = {"qblk": qblk, "qsqm": qsqm, "ddm": ddm}
        bad = []
        for name, mine in got.items():
            theirs = np.array(ref[f"{key}::{name}"])
            if mine.shape != theirs.shape:
                bad.append(f"{name}: форма {mine.shape} против {theirs.shape}")
            elif not bool((mine.view(np.uint8) == theirs.view(np.uint8)).all()):
                n_diff = int((mine.view(np.uint8) != theirs.view(np.uint8)).sum())
                bad.append(f"{name}: {n_diff} байт")
            n_bytes += mine.nbytes
        # xbits в манифесте сайдкара -- отдельное поле, сверим и его
        want_x = ref_manifest["tensors"][key].get("xbits")
        if want_x is not None and want_x != xbits:
            bad.append(f"xbits {xbits} против {want_x}")

        check(f"{key} [{OUT}x{IN}] xbits={xbits}", not bad, "; ".join(bad))

    print(f"\nсверено {len(picked)} тензоров ({len(seen)} форм), "
          f"{n_bytes / 1e6:.1f} МБ байт-в-байт")
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS else f"ПРОВАЛЕН: {len(FAILS)}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
