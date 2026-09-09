"""COMPRESSION: произвольное плечо, заданное JSON-спецификацией.

  python tests/_sess/ablate2.py <имя> '<json>'

JSON (все поля необязательны):
  {"bf16": true}                    -- эталон без квантования
  {"bits":      {"cmix": 5}}        -- битность группы
  {"overrides": {"ffn.key.weight": 6}}  -- битность ОТДЕЛЬНЫХ матриц
  {"modes":     {"cmix": "asym_sb6"}}   -- режим блочного квантования
  {"act": "/path/act_stats.pt"}     -- иная статистика для AW

По умолчанию -- пресет COMPRESSION со статистикой ACT_DEFAULT.
Одно плечо = один процесс (закон 2). Корпус мультиязычный, 38 окон
(закон 9). Чекпоинт грузится с mmap=True (закон 13).

ОГРАНИЧЕНИЕ writer: при asym_sb6* допустимы биты 4/5/6, иначе
NotImplementedError (падает громко). bits=16 отключает gw-ветку целиком.
"""
import copy, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import mlx.core as mx                                       # noqa: E402

from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.presets import COMPRESSION                   # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor        # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint    # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
ACT_DEFAULT = os.path.expanduser(
    "~/.cache/rwkv-quant/act_stats/act_12dbb2d3ac61eb20.pt")
OUT = os.path.expanduser("~/Develop/WKV-kvant/ablate_compression_fine.jsonl")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

FIELDS = ("dense", "codes", "codes_packed", "scale", "gw_qsqm", "gw_d",
          "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
          "outlier_indices", "outlier_values")


def build_cfg(spec):
    if spec.get("bf16"):
        return QuantConfig()
    cfg = copy.deepcopy(COMPRESSION)
    cfg.act_stats_path = os.path.expanduser(spec.get("act") or ACT_DEFAULT)
    if not os.path.exists(cfg.act_stats_path):
        raise SystemExit("нет статистики: " + cfg.act_stats_path)
    for g, b in spec.get("bits", {}).items():
        if g not in cfg.bits:
            raise SystemExit("нет такой группы: " + g)
        cfg.bits[g] = b
    for g, m in spec.get("modes", {}).items():
        if g not in cfg.group_scale_mode:
            raise SystemExit("у группы нет gw-режима: " + g)
        cfg.group_scale_mode[g] = m
    cfg.bits_overrides = dict(spec.get("overrides", {}))
    return cfg


def tensor_bytes(qt):
    n = 0
    for f in FIELDS:
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


def agg(per_seq, langs):
    res = {}
    for k in sorted(set(langs)) + ["ALL"]:
        sel = [x for x, l in zip(per_seq, langs) if k == "ALL" or l == k]
        res[k] = float(np.exp(sum(s for s, _ in sel) / sum(n for _, n in sel)))
    return res


def main():
    name = sys.argv[1]
    spec = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    cfg = build_cfg(spec)
    t0 = time.time()

    blob = torch.load(CORPUS)
    data, langs = blob["tokens"].numpy(), blob["lang"]
    sd = torch.load(CKPT, map_location="cpu", mmap=True)

    ov = spec.get("overrides", {})
    hit = [k for k in sd if any(p in k for p in ov)] if ov else []
    if ov and not hit:
        raise SystemExit(f"{name}: ни один ключ не совпал с {tuple(ov)}")

    tensors = {k: quantize_tensor(k, w, cfg, real_gw=True)
               for k, w in sd.items()}
    del sd
    ckpt = QuantizedCheckpoint(
        naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD, head_size=HEAD_SIZE,
        vocab_size=VOCAB, config_repr=repr(cfg), tensors=tensors)
    mb = sum(tensor_bytes(q) for q in ckpt.tensors.values()) / 1e6
    bits_hit = sorted({ckpt.tensors[k].bits for k in hit}) if hit else []

    model = QuantRWKV7(ckpt)
    res = agg(nll_per_seq(model, data), langs)
    rec = {"arm": name, "spec": spec, "size_mb": round(mb, 2), "ppl": res,
           "n_cut": len(hit), "bits_cut": bits_hit,
           "act": os.path.basename(getattr(cfg, "act_stats_path", "") or ""),
           "sec": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
