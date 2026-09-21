"""ПРОФИЛЬ ВХОДА o_proj ПО СЛОЯМ (20.09) -- кандидаты правила «какой слой
o_proj держать в высокой точности». LOO показал: энергия входа сама по
себе не предсказывает вклад (2.9B слой 31: max/med 7e4 -> 0; слой 0: 7e6
-> 85% вклада o). Поэтому рядом с энергией считается ОТНОСИТЕЛЬНАЯ
ОШИБКА ВЫХОДА слоя при реальном COMPRESSION-кванте (диагональное
приближение, x-каналы независимы):
    rel = sum_c E[x_c^2] ||dW[:,c]||^2 / sum_c E[x_c^2] ||W[:,c]||^2
Без прогона модели: веса + act-статистика.
    python oproj_profile_2009.py <чекпоинт> [act_stats.pt] [json]
Без act_stats -- сбор штатным collect() (кеш ~/.cache/rwkv-quant)."""
import copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # корень репо: Mac и сервер
import torch
from rwkv_quant.presets import COMPRESSION
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import dequantize_banded
from rwkv_quant.calibration import act_stats as A

TOK = os.environ.get("RWKVQ_TOK") or next(p for p in (
    "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt",
    os.path.expanduser("~/rwkvq/rwkv_vocab_v20230424.txt")) if os.path.exists(p))
ck = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] != "-":
    ap = sys.argv[2]
else:
    _, sig = A.collect(ck, TOK, verbose=False); ap = "%s/act_%s.pt" % (A.CACHE_DIR, sig)
st = torch.load(ap, map_location="cpu")
cfg = copy.deepcopy(COMPRESSION); cfg.act_stats_path = ap
sd = torch.load(ck, map_location="cpu", mmap=True)
keys = sorted((k for k in sd if k.endswith("att.output.weight")), key=lambda k: int(k.split(".")[1]))
rows = []
for k in keys:
    W = sd[k].float(); ex2 = st[k].float().flatten()
    q = quantize_tensor(k, sd[k], cfg, real_gw=True)
    dW = dequantize_banded(q, dtype=torch.float32) - W
    num = (ex2 * dW.pow(2).sum(0)).sum().item(); den = (ex2 * W.pow(2).sum(0)).sum().item()
    s = ex2.sort(descending=True).values
    rows.append(dict(layer=int(k.split(".")[1]), rel=num / den,
                     maxmed=(s[0] / ex2.median()).item(), top8=(s[:8].sum() / s.sum()).item(),
                     argmax=int(ex2.argmax()), bits=q.bits))
nl = len(rows)
print("%s  (%d слоёв, n_embd %d, стат. %s)" % (ck.split("/")[-1], nl, sd[keys[0]].shape[1], ap.split("/")[-1]))
print(" слой  rel_err   max/med   top8  канал")
med = sorted(r["rel"] for r in rows)[nl // 2]
for r in rows:
    print("%4d  %.5f %s %9.0f  %.2f  %5d" % (r["layer"], r["rel"], "*" if r["rel"] > 3 * med else " ",
                                             r["maxmed"], r["top8"], r["argmax"]))
print("медиана rel %.5f; max rel/медиана %.1f" % (med, max(r["rel"] for r in rows) / med))
if len(sys.argv) > 3:
    json.dump(rows, open(sys.argv[3], "w"))
