"""ПОЧЕМУ COMPRESSION НА 2.9B ХУЖЕ, ЧЕМ НА 1.5B (19.09): leave-one-out по
группам на РЕАЛЬНОМ пути, KL против fp32-эталона и ppl на 38 окнах.

Плечо = пресет COMPRESSION как он есть (включая поправку cmix.key -> 5), но
одна группа возвращена в bf16 (bits=16 -> gw-ветка не включается, разрез
чистый). Насколько упал KL -- столько группа и уносила. Сравнение ДОЛЕЙ групп
на 1.5B и 2.9B отвечает, какая группа переносится хуже.

Контроли: (1) base обязан воспроизвести KL файла на диске тем же рецептом
(1.5B 0.035989, 2.9B 0.052052); (2) у задетых ключей битность 16, и хотя бы
один ключ задет; (3) ключи вне разреза -- той же битности, что в base.
Одно плечо на процесс. Пер-окно KL/CE пишутся для парного бутстрэпа.
    RWKVQ_CKPT=... RWKVQ_ACT_STATS=... RWKVQ_KL_REF=... RWKVQ_OUT=... \
        python loo_compression_kl_1909.py <плечо>
"""
import copy, json, os, re, sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant"); sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests")
import numpy as np
import torch
import mlx.core as mx
from rwkv_quant.presets import COMPRESSION
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.schema import QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from ablate_subgroups import kl_stats

CKPT = os.environ["RWKVQ_CKPT"]; ACT = os.environ["RWKVQ_ACT_STATS"]
REF = os.environ["RWKVQ_KL_REF"]; OUT = os.environ["RWKVQ_OUT"]
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
CUTS = {
    "r": ("att.receptance.weight",), "k": ("att.key.weight",),
    "v": ("att.value.weight",), "o": ("att.output.weight",),
    "ffn_k": ("ffn.key.weight",), "ffn_v": ("ffn.value.weight",),
    "head": ("head.weight",), "emb": ("emb.weight",),
    "lora": ("att.w1", "att.w2", "att.a1", "att.a2", "att.v1", "att.v2", "att.g1", "att.g2"),
}
# ПЛЕЧИ ПРАВКИ (20.09): группа не в bf16, а в BITS[arm] бит -- премисса
# «o_proj в 5 битах» (раздел 19.09, НОЧЬ, шаг 3).
# o4s: те же 4 бита, но режим asym_sb6_search (поиск БЕЗ AW-статистики)
# только у задетых ключей -- отделяет «мало бит» от «AW портит o_proj».
UP = {"o5": ("att.output.weight",), "o6": ("att.output.weight",),
      "o4s": ("att.output.weight",)}
BITS = dict({a: 16 for a in CUTS}, o5=5, o6=6, o4s=4)
MODE = {"o4s": "asym_sb6_search"}
CUTS = dict(CUTS, **UP)
arm = sys.argv[1]
# ПОСЛОЙНЫЕ ПЛЕЧИ (20.09): oL<слои через +> -- o_proj ТОЛЬКО этих слоёв в
# bf16. Премисса: вклад o_proj 2.9B сидит в слоях с массивными каналами
# входа (слой 0: max/med E[x²] 7.2e6, слой 31: 7.2e4). Ключ с точкой после
# номера: "blocks.1.att.output" не задевает blocks.1x.
if arm.startswith("oL"):
    CUTS[arm] = tuple("blocks.%d.att.output.weight" % int(i) for i in arm[2:].split("+"))
    BITS[arm] = 16
assert arm == "base" or arm in CUTS, "плечи: base " + " ".join(CUTS)
TB = BITS.get(arm, 16)
cfg = copy.deepcopy(COMPRESSION); cfg.act_stats_path = ACT
cut = CUTS.get(arm, ())
# разрез ПЕРВЫМ: совпадение ищется подстрокой, первое побеждает
# поправки пресета, НЕ перекрытые разрезом (иначе ffn.key: 16 затирается
# штатным ffn.key: 5 -- поймано контролем битности 19.09)
cfg.bits_overrides = dict({p: TB for p in cut},
                          **{k: v for k, v in COMPRESSION.bits_overrides.items() if k not in cut})
t0 = time.time()
sd = torch.load(CKPT, map_location="cpu", mmap=True)
hit = [k for k in sd if any(p in k for p in cut)]
if cut and not hit:
    raise SystemExit("плечо %s: ни один ключ не задет" % arm)
cfg_cut = cfg
if arm in MODE:
    cfg_cut = copy.deepcopy(cfg)
    cfg_cut.group_scale_mode = dict(cfg.group_scale_mode, proj=MODE[arm])
tensors = {k: quantize_tensor(k, w, cfg_cut if k in hit else cfg, real_gw=True)
           for k, w in sd.items()}
n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
n_embd = int(sd["emb.weight"].shape[1])
del sd
bits_hit = sorted({tensors[k].bits for k in hit})
if cut and bits_hit != [TB]:
    raise SystemExit("плечо %s: задетые ключи не %d бит: %s" % (arm, TB, bits_hit))
bits_map = {k: q.bits for k, q in tensors.items()}
# байты полезной нагрузки: сумма всех тензорных полей (для разности плеч)
nbytes = sum(v.numel() * v.element_size() for q in tensors.values()
             for v in vars(q).values() if isinstance(v, torch.Tensor))
ckpt = QuantizedCheckpoint(naming="world", n_layer=n_layer, n_embd=n_embd, head_size=64,
                           vocab_size=65536, config_repr=repr(cfg), tensors=tensors)
model = QuantRWKV7(ckpt)
del tensors, ckpt
blob = torch.load(CORPUS)
data = blob["tokens"][:, :512].numpy(); langs = list(blob["lang"])
ref = np.load(REF, mmap_mode="r")
kl_s, ce_s, ce_r, top = [], [], [], []
for i in range(ref.shape[0]):
    lg = model(mx.array(data[i:i + 1, :-1])).astype(mx.float32); mx.eval(lg)
    got = np.array(lg)[0]; del lg
    R = np.asarray(ref[i]); tgt = data[i, 1:512]
    k, t = kl_stats(R, got)
    kl_s.append(float(k.mean())); top.append(float(t.mean()))
    for arr, dst in ((got, ce_s), (R, ce_r)):
        a = arr.astype(np.float64); m = a.max(1)
        dst.append(float((m + np.log(np.exp(a - m[:, None]).sum(1)) - a[np.arange(len(tgt)), tgt]).mean()))
doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
doc[arm] = {"kl": kl_s, "ce": ce_s, "ce_ref": ce_r, "top1": top, "langs": langs,
            "n_cut": len(hit), "bits": bits_map, "nbytes": nbytes, "sec": round(time.time() - t0)}
json.dump(doc, open(OUT, "w"))
print("%s: KL %.6f  Δppl %+.3f%%  top-1 %.3f%%  задето %d  %.1f МБ  %d с" % (
    arm, np.mean(kl_s), 100 * (np.exp(np.mean(ce_s)) / np.exp(np.mean(ce_r)) - 1),
    100 * np.mean(top), len(hit), nbytes / 1e6, time.time() - t0), flush=True)
