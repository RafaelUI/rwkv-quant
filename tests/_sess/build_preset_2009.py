"""Сборка пресета публичным путём (api.quantize), act_stats -- авто-кеш.
    python build_preset_2009.py <чекпоинт> <пресет> <выход.rwkvq>"""
import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
from rwkv_quant.api import quantize
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
t0 = time.time()
name = sys.argv[2]
if name.endswith("_nol0"):
    # контрольная сборка: тот же пресет БЕЗ o_proj слоя 0 в bf16
    import copy
    from rwkv_quant.presets import PRESETS, O_PROJ_L0_BF16
    cfg = copy.deepcopy(PRESETS[name[:-5]])
    cfg.bits_overrides = {k: v for k, v in cfg.bits_overrides.items() if k not in O_PROJ_L0_BF16}
    quantize(sys.argv[1], sys.argv[3], config=cfg, tokenizer=TOK, verbose=False)
else:
    quantize(sys.argv[1], sys.argv[3], preset=name, tokenizer=TOK, verbose=False)
print("готово: %s  %d с" % (sys.argv[3], time.time() - t0), flush=True)
