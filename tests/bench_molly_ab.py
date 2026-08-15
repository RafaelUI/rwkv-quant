"""НАША РАСКЛАДКА ПРОТИВ ШТАТНОГО MLX-AFFINE: pp512 и tg128 чередованием.

ЗАЧЕМ ИМЕННО ЭТОТ ЗАМЕР. Открытый вопрос про префилл упирается в
квантованный GEMM: репак нашей раскладки в родной `mx.quantized_matmul`.
Цена этой инженерии неизвестна, а вот ЦЕНА ЕЁ ОТСУТСТВИЯ измерима прямо
сейчас -- чекпоинт MollySophia квантован штатным `mx.nn.quantize`
(affine, group_size=64, bits=6) и потому целиком идёт через
`mx.quantized_matmul`. То есть он и есть «мы после репака», только без
нашей сетки кодов и без AW.

ПОЧЕМУ СРАВНЕНИЕ ЧЕСТНОЕ ПО СКОРОСТИ И НЕЧЕСТНОЕ ПО КАЧЕСТВУ. Формы
совпадают полностью (hidden 2048, 24 слоя, 32 головы по 64, vocab 65536,
ранги 96/96/64/256), бит/вес тоже: affine gs=64 при шести битах стоит
6 + 2*16/64 = 6.5 -- ровно как наш sb6. Но ЧЕКПОИНТ ДРУГОЙ (g1g против
нашего g1h), поэтому ppl между ними не сопоставим в принципе, и здесь он
не меряется. Меряется только время.

Обе модели живут в памяти ОДНОВРЕМЕННО и бёрсты чередуются (закон 1):
без этого дрейф безвентиляторной машины съест разницу. Своп печатается на
границе и по раундам -- рост во время замера делает его недействительным
(закон 11).

    RWKVQ_MOLLY_DIR=... python tests/bench_molly_ab.py [our.rwkvq] [раундов]
"""
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

MOLLY = os.environ.get("RWKVQ_MOLLY_DIR", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-1.5B-g1g-ctx8192-mlx-6bit"))
OURS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/reduction_sym_head8.rwkvq"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PP, TG = 512, 64

# Трафик за токен, посчитанный ПО ФОРМУЛЕ бит/вес, а не по файлу: `emb` --
# это gather одной строки, её в трафик включать нельзя (закон 12).
#   квантованные линейные: 24 слоя x (4 x [2048,2048] + [8192,2048] +
#   [2048,8192]) = 1208M весов, плюс голова 134.2M.
#   LoRA-ветки в обеих моделях лежат плотным fp16.
_QW_LAYERS, _QW_HEAD, _LORA_MB = 1208.0e6, 134.2e6, 100.1


def traffic_mb(bits_body, bits_head):
    return ((_QW_LAYERS * bits_body + _QW_HEAD * bits_head) / 8 / 1e6
            + _LORA_MB)


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def bench_pp(model, prompt):
    st = model.init_state(1)
    logits, st = model.step(prompt, st, True)
    mx.eval(logits)
    mx.synchronize()
    t0 = time.perf_counter()
    st = model.init_state(1)
    logits, st = model.step(prompt, st, True)
    mx.eval(logits)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3, logits


def bench_tg(model, logits, n=TG):
    st = model.init_state(1)
    _, st = model.step(mx.array(np.array([[1]], dtype=np.int32)), st)
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(8):
        lg, st = model.step(tok[None], st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        lg, st = model.step(tok[None], st)
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok)
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    import eval_molly_real as molly
    molly.MODEL_DIR = MOLLY
    print(f"наша модель: {OURS}\nштатный MLX: {MOLLY}", flush=True)

    ours = qm.QuantRWKV7(load_raw(OURS))
    w = mx.load(f"{MOLLY}/model.safetensors")
    theirs = molly.MollyRWKV7(w)
    del w
    gc.collect()
    mx.clear_cache()

    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(1, 60000, size=(1, PP)).astype(np.int32))

    models = [("наша (sym+head@8)", ours), ("штатный mlx-affine", theirs)]
    for _, m in models:                      # прогрев обеих трассировок
        _, lg = bench_pp(m, prompt)
        bench_tg(m, lg, 8)

    sw0 = swap_mb()
    res = {lab: {"pp": [], "tg": []} for lab, _ in models}
    for r in range(ROUNDS):
        for lab, m in models:                # ЧЕРЕДОВАНИЕ
            t_pp, lg = bench_pp(m, prompt)
            res[lab]["pp"].append(t_pp)
            res[lab]["tg"].append(bench_tg(m, lg))
        print(f"  раунд {r+1}/{ROUNDS}, своп {swap_mb():.0f} МБ", flush=True)
    sw1 = swap_mb()

    print(f"\n{'модель':>20} | {'pp512':>10} | {'tg':>9} | {'файл':>8} | "
          f"{'трафик/ток':>10} | {'ГБ/с декода':>11}")
    traf = {"наша (sym+head@8)": traffic_mb(6.5625, 8.5625),
            "штатный mlx-affine": traffic_mb(6.5, 6.5)}
    size = {"наша (sym+head@8)": os.path.getsize(OURS) / 1e6,
            "штатный mlx-affine":
                os.path.getsize(f"{MOLLY}/model.safetensors") / 1e6}
    for lab, _ in models:
        pp = float(np.median(res[lab]["pp"]))
        tg = float(np.median(res[lab]["tg"]))
        spp = 100 * (max(res[lab]["pp"]) - min(res[lab]["pp"])) / pp
        stg = 100 * (max(res[lab]["tg"]) - min(res[lab]["tg"])) / tg
        print(f"{lab:>20} | {PP/pp*1e3:7.1f} т/с | {1000/tg:6.1f} т/с | "
              f"{size[lab]:7.1f} | {traf[lab]:9.1f} | "
              f"{traf[lab]/tg:10.1f}")
        print(f"{'':>20} | разброс {spp:.1f}% | {stg:.1f}%")
    print(f"\nсвоп: {sw0:.0f} -> {sw1:.0f} МБ (дельта {sw1-sw0:+.0f}; "
          f"ненулевая делает замер недействительным)")
    print("ppl тут НЕ меряется: чекпоинты разные (g1g против g1h), "
          "качество между ними не сопоставимо в принципе.")


if __name__ == "__main__":
    main()
