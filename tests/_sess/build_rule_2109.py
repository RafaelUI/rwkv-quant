"""Сборка .rwkvq по правилу "слабое место -> бит" (21.09) публичным путём.
База -- COMPRESSION в состоянии 632eee3 БЕЗ o_proj слоя 0 в bf16 и с emb 5
(то, на чём мерились лестницы: compression_nol0 в ablate_sym_composite),
поверх -- точные ключи правила (ставятся ПЕРВЫМИ: подстрока, первое побеждает,
иначе "ffn.key.weight": 5 перебил бы шаг правила на ffn.key).
    python build_rule_2109.py <чекпоинт> <ovr.json | -> <выход.rwkvq>
"""
import copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant.api import quantize
from rwkv_quant.presets import PRESETS, O_PROJ_L0_BF16
TOK = os.environ.get("RWKVQ_TOK") or next(p for p in (
    "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt",
    os.path.expanduser("~/rwkvq/rwkv_vocab_v20230424.txt")) if os.path.exists(p))
t0 = time.time()
cfg = copy.deepcopy(PRESETS["compression"])
cfg.bits = dict(cfg.bits, emb=5)
base_ovr = {k: v for k, v in cfg.bits_overrides.items() if k not in O_PROJ_L0_BF16}
ovr = {} if sys.argv[2] == "-" else json.load(open(sys.argv[2]))
cfg.bits_overrides = dict(ovr, **{k: v for k, v in base_ovr.items() if k not in ovr})
print("переопределений правила: %d; база: %s" % (len(ovr), base_ovr), flush=True)
quantize(sys.argv[1], sys.argv[3], config=cfg, tokenizer=TOK, verbose=False)
print("готово: %s  %.1f МБ  %d с" % (sys.argv[3], os.path.getsize(sys.argv[3]) / 1e6, time.time() - t0), flush=True)
