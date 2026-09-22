"""Сборка .rwkvq по правилу ОТ ПРЕСЕТА (вариант (а), 22.09): база -- COMPRESSION
как есть (emb 6, o_proj слоя 0 bf16, ffn.key 5), поверх -- точные ключи правила.
Ключи правила ставятся ПЕРВЫМИ: bits_overrides сопоставляется подстрокой, первое
совпадение побеждает -- иначе "ffn.key.weight": 5 перебил бы шаг на ffn.key.
    python build_rule_a_2209.py <чекпоинт> <ovr.json> <выход.rwkvq>
"""
import copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant.api import quantize
from rwkv_quant.presets import PRESETS
TOK = os.environ.get("RWKVQ_TOK") or next(p for p in (
    "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt",
    os.path.expanduser("~/rwkvq/rwkv_vocab_v20230424.txt")) if os.path.exists(p))
t0 = time.time()
cfg = copy.deepcopy(PRESETS["compression"])
ovr = json.load(open(sys.argv[2]))
base = dict(cfg.bits_overrides)
cfg.bits_overrides = dict(ovr, **{k: v for k, v in base.items() if k not in ovr})
print("правило: %d ключей; база пресета: bits %s, overrides %s" % (len(ovr), cfg.bits, base), flush=True)
quantize(sys.argv[1], sys.argv[3], config=cfg, tokenizer=TOK, verbose=False)
print("готово: %s  %.2f МБ  %d с" % (sys.argv[3], os.path.getsize(sys.argv[3]) / 1e6, time.time() - t0), flush=True)
