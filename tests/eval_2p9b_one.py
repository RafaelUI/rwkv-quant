"""
Один конфиг на 2.9B за один запуск процесса. Результат дописывается в
общий JSON, таблица собирается отдельно (tests/eval_2p9b_report.py).

Почему по одному процессу на конфиг: на 1.5B прогон всех конфигов в одном
процессе загонял машину в своп на 5 ГБ (RSS при этом показывал 0.6 ГБ --
MLX держит массивы в unified memory, и ps их не видит; смотреть надо
vm_stat / vm.swapusage). На 2.9B веса вдвое больше, накопление между
конфигами машину на 16 ГБ просто убьёт. Процесс завершился -- вся память
вернулась системе гарантированно, без надежд на gc и mx.clear_cache.

Прочие меры против свопа:
  - torch.load(mmap=True): тензоры чекпоинта остаются file-backed
    страницами, ядро вытесняет их само, а не через своп;
  - bf16-базлайн НЕ идёт через quantize_tensor: там на bits>=16 стоит
    .clone().contiguous(), то есть честные 5.9 ГБ анонимной памяти
    поверх уже отображённого файла. Держим ссылку на mmap-тензор;
  - ckpt удаляется сразу после постройки модели: дальше нужны только
    mx-массивы.

Запуск: python tests/eval_2p9b_one.py <config>
  config: bf16 | reduction | reduction_fix | compression | compression_fix
"""
import copy
import gc
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mlx.core as mx  # noqa: E402

from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.presets import REDUCTION, COMPRESSION  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402
from rwkv_quant.formats.schema import QuantizedTensor, QuantizedCheckpoint  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

CKPT_PTH = os.path.expanduser("~/Develop/rwkv7-g1h-2.9b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
OUT_JSON = os.path.expanduser("~/Develop/WKV-kvant/eval_2p9b.json")
ACT_STATS = "/tmp/act_stats_2p9b_multiling.pt"
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 32, 2560, 64, 65536


def mem(tag):
    """Системные цифры: ps/RSS не видит unified memory MLX."""
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True).stdout.strip()
    mp = subprocess.run(["memory_pressure", "-Q"],
                        capture_output=True, text=True).stdout
    free = [l for l in mp.splitlines() if "free percentage" in l]
    print(f"  [mem/{tag}] {sw} | {free[0].strip() if free else ''}", flush=True)


def tweak(preset, small=None, proj_mode=None):
    cfg = copy.deepcopy(preset)
    cfg.act_stats_path = ACT_STATS
    if small is not None:
        cfg.bits["small"] = small
    if proj_mode is not None:
        cfg.group_scale_mode = dict(cfg.group_scale_mode, proj=proj_mode)
    return cfg


def get_config(name):
    return {
        "bf16": lambda: QuantConfig(),
        "reduction": lambda: tweak(REDUCTION),
        # reduction_fix меняет ДВЕ вещи сразу; на 2.9B сербский от него
        # ухудшился, поэтому нужен вариант с одним изменением, чтобы
        # понять, какое из двух виновато
        "reduction_small": lambda: tweak(REDUCTION, small=16),
        "reduction_fix": lambda: tweak(REDUCTION, small=16,
                                       proj_mode="asym_sb6_search"),
        "compression": lambda: tweak(COMPRESSION),
        "compression_fix": lambda: tweak(COMPRESSION, small=16),
    }[name]()


def tensor_bytes(qt):
    n = 0
    for f in ("dense", "codes", "codes_packed", "scale", "gw_qsqm", "gw_d",
              "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
              "outlier_indices", "outlier_values"):
        t = getattr(qt, f, None)
        if t is not None:
            n += t.numel() * t.element_size()
    return n


def nll_per_seq(model, data):
    out = []
    for i in range(data.shape[0]):
        b = data[i:i + 1]
        logits = model(mx.array(b[:, :-1])).astype(mx.float32)
        tgt = mx.array(b[:, 1:])[..., None]
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, tgt, axis=-1).squeeze(-1)
        mx.eval(nll)
        a = np.array(nll)
        out.append((float(a.sum()), int(a.size)))
        del logits, nll
    return out


def main():
    name = sys.argv[1]
    cfg = get_config(name)
    blob = torch.load(CORPUS)
    data, langs = blob["tokens"].numpy(), blob["lang"]
    print(f"=== 2.9B / {name} ===", flush=True)
    mem("старт")

    t0 = time.time()
    sd = torch.load(CKPT_PTH, map_location="cpu", mmap=True)
    print(f"  mmap-загрузка {len(sd)} тензоров за {time.time()-t0:.0f}s", flush=True)
    mem("после load")

    t0 = time.time()
    if name == "bf16":
        # без .clone() -- страницы остаются file-backed
        tensors = {k: QuantizedTensor(key=k, group="other", bits=16,
                                      shape=tuple(w.shape),
                                      dense=w if w.dtype == torch.bfloat16
                                      else w.to(torch.bfloat16))
                   for k, w in sd.items()}
    else:
        tensors = {k: quantize_tensor(k, w, cfg, real_gw=True)
                   for k, w in sd.items()}
    mb = sum(tensor_bytes(q) for q in tensors.values()) / 1e6
    print(f"  квантование {time.time()-t0:.0f}s, {mb:.2f} MB", flush=True)
    mem("после квантования")

    ckpt = QuantizedCheckpoint(naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD,
                               head_size=HEAD_SIZE, vocab_size=VOCAB,
                               tensors=tensors, config_repr=repr(cfg))
    model = QuantRWKV7(ckpt)
    del ckpt, tensors, sd
    gc.collect()
    mem("после постройки модели")

    t0 = time.time()
    per_seq = nll_per_seq(model, data)
    res = {}
    for k in sorted(set(langs)) + ["ALL"]:
        sel = [x for x, l in zip(per_seq, langs) if k == "ALL" or l == k]
        res[k] = float(np.exp(sum(s for s, _ in sel) / sum(n for _, n in sel)))
    print(f"  ppl {time.time()-t0:.0f}s: "
          + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
    mem("после ppl")

    all_res = {}
    if os.path.exists(OUT_JSON):
        all_res = json.load(open(OUT_JSON))
    all_res[name] = {"size_mb": mb, "ppl": res}
    json.dump(all_res, open(OUT_JSON, "w"), indent=2)
    print(f"  -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
