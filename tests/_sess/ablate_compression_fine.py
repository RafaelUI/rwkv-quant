"""COMPRESSION: помотрицное leave-one-out по телу и голове.

Один компонент возвращается в bf16, всё остальное -- ровно пресет
COMPRESSION. Насколько упал Dppl -- столько компонент и уносил.

Одно плечо = один процесс (закон 2). Корпус мультиязычный, 38 окон
(закон 9). act_stats -- честная, с репозиторного calib_corpus, снята
07.09 и не пересекается с корпусом ppl.

bits=16 отключает gw-ветку целиком (writer идёт в gw только при
bits < 16), поэтому разрез чистый: не смена раскладки, а отсутствие
квантования у одной матрицы.

Запуск: python tests/_sess/ablate_compression_fine.py <плечо>
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
ACT = os.path.expanduser(
    "~/.cache/rwkv-quant/act_stats/act_12dbb2d3ac61eb20.pt")
OUT = os.path.expanduser("~/Develop/WKV-kvant/ablate_compression_fine.jsonl")
NAMING, N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = "world", 24, 2048, 64, 65536

# Подстроки кроют ОБА именования: world (att./ffn.) и внутреннее
# (_proj/cmix.). Совпадение ищется как подстрока ключа, первое побеждает.
CUTS = {
    "r":     ("att.receptance.weight", "r_proj.weight"),
    "k":     ("att.key.weight", "k_proj.weight"),
    "v":     ("att.value.weight", "v_proj.weight"),
    "o":     ("att.output.weight", "o_proj.weight"),
    "ffn_k": ("ffn.key.weight", "cmix.key.weight"),
    "ffn_v": ("ffn.value.weight", "cmix.value.weight"),
    "head":  ("head.weight",),
    "emb":   ("emb.weight",),
}
ARMS = ["bf16", "base"] + list(CUTS)

FIELDS = ("dense", "codes", "codes_packed", "scale", "gw_qsqm", "gw_d",
          "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
          "outlier_indices", "outlier_values")


def build_cfg(arm):
    if arm == "bf16":
        return QuantConfig(), {}
    cfg = copy.deepcopy(COMPRESSION)
    cfg.act_stats_path = ACT
    if arm == "base":
        cfg.bits_overrides = {}
        return cfg, {}
    ov = {p: 16 for p in CUTS[arm]}
    cfg.bits_overrides = ov
    return cfg, ov


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
    arm = sys.argv[1]
    if arm not in ARMS:
        raise SystemExit("плечи: " + " ".join(ARMS))
    cfg, ov = build_cfg(arm)
    t0 = time.time()

    blob = torch.load(CORPUS)
    data, langs = blob["tokens"].numpy(), blob["lang"]
    sd = torch.load(CKPT, map_location="cpu")

    # КОНТРОЛЬ, что плечо действительно отличается: разрез обязан задеть
    # хотя бы один тензор, иначе это молча то же самое, что base.
    hit = [k for k in sd if any(p in k for p in ov)] if ov else []
    if ov and not hit:
        raise SystemExit(f"плечо {arm}: ни один ключ не совпал с {tuple(ov)}")

    tensors = {k: quantize_tensor(k, w, cfg, real_gw=True)
               for k, w in sd.items()}
    del sd
    ckpt = QuantizedCheckpoint(
        naming=NAMING, n_layer=N_LAYER, n_embd=N_EMBD, head_size=HEAD_SIZE,
        vocab_size=VOCAB, config_repr=repr(cfg), tensors=tensors)
    mb = sum(tensor_bytes(q) for q in ckpt.tensors.values()) / 1e6
    # второй контроль: у задетых ключей битность обязана быть 16
    bits_hit = sorted({ckpt.tensors[k].bits for k in hit}) if hit else []

    model = QuantRWKV7(ckpt)
    res = agg(nll_per_seq(model, data), langs)
    rec = {"arm": arm, "size_mb": round(mb, 2), "ppl": res,
           "n_cut": len(hit), "bits_cut": bits_hit,
           "sec": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
