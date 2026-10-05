"""06.10, находка 2: подпись кеша act_stats = ... + type(tokenizer).__name__. Воспроизведение обоих следствий на 0.1B.
Кеш подменён на временный каталог -- рабочий ~/.cache/rwkv-quant не трогается. Колесо /tmp/rq_site_p1, cwd /tmp."""
import json, os, sys, time
import torch
import rwkv_quant
assert "/tmp/rq_site_p1" in rwkv_quant.__file__
from rwkv_quant.calibration import act_stats as A
from rwkv_quant.formats import codec
W = os.path.expanduser("~/Develop/WKV-kvant/"); CK = W + "rwkv7-g1d-0.1b.pth"
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
print("рабочий кеш:", A.CACHE_DIR)
tmp = "/tmp/rq_p2_%d" % int(time.time()); os.makedirs(tmp); A.CACHE_DIR = tmp   # временный кеш; убирать в Корзину руками
enc = A._encoder(TOK)
class T:
    def encode(self, s): return enc(s)
# (а) один словарь, три формы -> три ключа
sigs = {n: A._signature(CK, t, A.CORPUS) for n, t in (("путь", TOK), ("функция", enc), ("объект T", T()))}
print("(а) один словарь, три формы:", sigs, "-> разных ключей", len(set(sigs.values())))
# (б) два РАЗНЫХ словаря путём -> один ключ (подпись не читает файл)
tok2 = os.path.join(tmp, "other_vocab.txt")
print("(б) подпись для несуществующего/другого пути:", A._signature(CK, tok2, A.CORPUS), "== подписи настоящего:", A._signature(CK, tok2, A.CORPUS) == sigs["путь"])
# (б') сквозная подстановка: два РАЗНЫХ токенизатора-функции -> второй получает статистику первого из кеша
def bytes_tok(s): return list(s.encode("utf-8"))          # другой словарь: байты, id < 256
def world_fn(s): return enc(s)                         # тот же мировой словарь, но обычной функцией (type -> "function", как у bytes_tok)
print("имена типов: путь %r, enc %r, world_fn %r, bytes_tok %r" % (type(TOK).__name__, type(enc).__name__, type(world_fn).__name__, type(bytes_tok).__name__))
s1, g1 = A.collect(CK, world_fn, verbose=False)
s2, g2 = A.collect(CK, bytes_tok, verbose=False)           # кеш: вернёт чужое
s3, g3 = A.collect(CK, bytes_tok, verbose=False, cache=False)   # честный пересчёт байтовым словарём
k = "blocks.5.ffn.key.weight" if "blocks.5.ffn.key.weight" in s1 else sorted(s1)[len(s1) // 2]
rel = lambda a, b: float(((a.float() - b.float()).abs().max() / a.float().abs().max()))
print("(б') ключи кеша: %s и %s (равны: %s); статистика '%s': из кеша для байтового == мировой словарь: %s; честная байтовая отличается на rel %.2f"
      % (g1, g2, g1 == g2, k, bool(torch.equal(s1[k], s2[k])), rel(s1[k], s3[k])))
# что в манифесте зависит от подписи / формы токенизатора
for f in ("red.rwkvq", "red_c.rwkvq", "red_o.rwkvq", "comp_default.rwkvq"):
    p = "/tmp/rq_smoke/" + f
    if not os.path.exists(p): continue
    man, arr = codec.open_rwkvq(p)
    cfgj = man.get("config") or {}
    print("%-20s tokenizer=%r\n%20s act_stats_path=%r\n%20s autopick.measure_signature=%r gptq=%r" % (
        f, man.get("tokenizer"), "", (cfgj.get("act_stats_path") if isinstance(cfgj, dict) else None), "",
        (man.get("autopick") or {}).get("measure_signature"), {k: v for k, v in (man.get("gptq") or {}).items() if k in ("calib", "calib_sha")}))
print("ключи манифеста:", sorted(man))
print("временный кеш:", tmp); print("ГОТОВО")
