"""04.10, СЕРВЕР: наши файлы РАВНОГО размера с MLX int6 и GGUF Q6_K -- сетка REDUCTION (sym, Q6_K-тип) с 6 битами:
  sym6    -- proj 8 -> 6, emb 8 -> 6, head 8 -> 6 (всё в 6 бит, как int6: 1.5B ~1271 МБ против 1272.4; 2.9B ~2450 против 2453.4)
  sym6h8  -- только proj 8 -> 6, emb / head остаются 8 (1.5B ~1338 МБ против Q6_K 1336.1)
Остальное -- как REDUCTION (o_proj слоя 0 в bf16, LoRA, cmix 6). GPTQ -- по RWKVQ_GPTQ (1 | 0), автоподбора нет.
    python make_cfg_files_0410.py <ckpt> <метка> <sym6|sym6h8> <тег>   -> ~/rwkvq/files_0310/<метка>_<тег>.rwkvq"""
import copy, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant import api, presets
CK, LAB, KIND, TAG = sys.argv[1:5]
TOK = os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
GP = os.environ.get("RWKVQ_GPTQ", "1") == "1"
c = copy.deepcopy(presets.REDUCTION)
assert c.bits["proj"] == 8 and c.bits["emb"] == 8 and c.bits["head"] == 8 and c.bits["cmix"] == 6, c.bits
c.bits["proj"] = 6
if KIND == "sym6":
    c.bits["emb"] = 6; c.bits["head"] = 6
else:
    assert KIND == "sym6h8", KIND
D = os.path.expanduser("~/rwkvq/files_0310"); p = os.path.join(D, "%s_%s.rwkvq" % (LAB, TAG))
print("старт %s %s %s тег %s gptq %s биты %s" % (time.ctime(), LAB, KIND, TAG, GP, c.bits), flush=True)
t0 = time.time()
api.quantize(CK, p, config=c, tokenizer=TOK, device=os.environ.get("RWKVQ_DEVICE"), verbose=True, autopick=False, gptq=GP)
print("ФАЙЛ %s %s %d байт, %.0f с" % (LAB, TAG, os.path.getsize(p), time.time() - t0), flush=True)
