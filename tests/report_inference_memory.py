"""
Что на самом деле лежит в памяти при инференсе против того, что на диске.

Вопрос не праздный: `.rwkvq` на 2.9B весит 1855 МБ, и легко решить, что
столько же займёт модель. Это неверно в обе стороны -- часть тензоров при
загрузке РАСШИРЯЕТСЯ, часть перекладывается, а часть дублируется фьюзом.

Обход только по __dict__, БЕЗ getattr по именам: у GwQuantLinear
старые имена (codes/qs/qm/d/dm/qh/qh2) -- ленивые view через __getattr__,
и обращение к ним материализовало бы буферы, которых в памяти нет,
превратив отчёт в самосбывающийся прогноз.

    python tests/report_inference_memory.py <model.rwkvq>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402

from rwkv_quant.formats.reader import load_raw  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

DISK_FIELDS = ("codes", "codes_packed", "scale", "dense", "gw_qsqm", "gw_d",
               "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
               "outlier_indices", "outlier_values")


def disk_breakdown(ckpt):
    by = {}
    for qt in ckpt.tensors.values():
        kind = "dense" if qt.bits >= 16 else (qt.gw_mode or "rtn")
        for f in DISK_FIELDS:
            t = getattr(qt, f, None)
            if t is not None:
                by[kind] = by.get(kind, 0) + t.numel() * t.element_size()
    return by


def walk(obj, path, seen, out, depth=0):
    """mx-массивы в объекте модели, с дедупликацией по id буфера."""
    if depth > 6:
        return
    if isinstance(obj, mx.array):
        key = id(obj)
        if key not in seen:
            seen.add(key)
            out.append((path, obj.size * obj.dtype.size, str(obj.dtype)))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]", seen, out, depth + 1)
        return
    d = getattr(obj, "__dict__", None)
    if not d:
        return
    for k, v in d.items():
        walk(v, f"{path}.{k}" if path else k, seen, out, depth + 1)


def bucket(path, dtype):
    p = path.split(".")[-1]
    if p in ("qblk", "qsqm", "ddm"):
        return "sb6 (K3-интерлив)"
    if "emb_weight" in path:
        return "emb"
    if "lora" in path.lower() or "wav_" in path or "_A" in p or "_B" in p:
        return f"LoRA-ветки {dtype}"
    if dtype == "float32":
        return "мелочь в fp32 (нормы, token-shift)"
    return f"прочее {dtype}"


def main():
    path = sys.argv[1]
    size_mb = os.path.getsize(path) / 1e6
    ckpt = load_raw(path)

    print(f"{os.path.basename(path)}: файл {size_mb:.1f} МБ")
    print("\nна диске:")
    disk = disk_breakdown(ckpt)
    for k, v in sorted(disk.items(), key=lambda x: -x[1]):
        print(f"  {k:<8} {v / 1e6:8.1f} МБ")
    print(f"  {'ИТОГО':<8} {sum(disk.values()) / 1e6:8.1f} МБ")

    mx.clear_cache()
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    model = QuantRWKV7(ckpt)
    mx.eval(model.emb_weight)
    active = (mx.get_active_memory() - before) / 1e6
    peak = mx.get_peak_memory() / 1e6
    mx.clear_cache()
    after_clear = (mx.get_active_memory() - before) / 1e6

    seen, arrays = set(), []
    walk(model, "", seen, arrays)
    by = {}
    for p, n, dt in arrays:
        b = bucket(p, dt)
        e = by.setdefault(b, [0, 0])
        e[0] += n
        e[1] += 1

    print("\nв памяти (обход модели):")
    for k, (v, n) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"  {k:<34} {v / 1e6:8.1f} МБ  ({n} массивов)")
    total = sum(v for v, _ in by.values()) / 1e6
    print(f"  {'ИТОГО':<34} {total:8.1f} МБ")

    print(f"\nаллокатор MLX: active {active:.1f} МБ, "
          f"после clear_cache {after_clear:.1f} МБ, пик за загрузку {peak:.1f} МБ")
    print(f"файл {size_mb:.1f} МБ -> резидентно {after_clear:.1f} МБ "
          f"({after_clear / size_mb:.2f}x), пик {peak / size_mb:.2f}x")

    print("\nтоп по расходу:")
    top = {}
    for p, n, dt in arrays:
        key = p.split(".")[-1] if "blocks" not in p else \
            "blocks[]." + ".".join(p.split(".")[2:])
        top[key] = top.get(key, 0) + n
    for k, v in sorted(top.items(), key=lambda x: -x[1])[:8]:
        print(f"  {v / 1e6:9.1f} МБ  {k}")


if __name__ == "__main__":
    main()
