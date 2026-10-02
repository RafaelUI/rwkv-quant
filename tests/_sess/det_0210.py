"""02.10, СЕРВЕР: детерминизм библиотечного GPTQ. G.run (600 окон корпуса пакета, damp 0.1, autopick из кеша как в API)
-> хеши упакованных тензоров; сравнение с тензорами УЖЕ ЗАПИСАННОГО файла (собран quantize() 01.10 под нагрузкой сервера).
Два запуска на разных картах + файл: совпадение всех трёх -- детерминизм; расхождение -- источник разброса GPTQ.
    python det_0210.py <ckpt> <reduction|compression> <файл.rwkvq> <метка>"""
import copy, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, autopick as ap, gptq as G
from rwkv_quant.formats.reader import load_raw
CK, PR, F, LAB = sys.argv[1:5]
TOK = os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
t0 = time.time()
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.REDUCTION if PR == "reduction" else presets.COMPRESSION)
cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
if PR == "compression":
    with open(F, "rb") as f:
        import struct; n = struct.unpack("<Q", f.read(8))[0]; man = json.loads(json.loads(f.read(n))["__metadata__"]["rwkvq"])
    cfg.bits_overrides = dict(man["config"]["bits_overrides"])
qts = G.run(CK, TOK, cfg, verbose=False, device="cuda")
h = lambda qt: hashlib.sha1(b"".join(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                                     for _, v in sorted(vars(qt).items()) if isinstance(v, torch.Tensor))).hexdigest()
mine = {k: h(v) for k, v in qts.items()}
fil = load_raw(F).tensors
diff = [k for k in mine if h(fil[k]) != mine[k]]
json.dump(mine, open(os.path.expanduser("~/rwkvq/det_%s.json" % LAB), "w"))
print("%s: %d GPTQ-тензоров, отличаются от файла: %d %s; %.0f с" % (LAB, len(mine), len(diff), diff[:5], time.time() - t0), flush=True)
