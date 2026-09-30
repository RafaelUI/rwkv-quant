"""30.09, СЕРВЕР: реальные размеры файлов для сводной таблицы -- api.quantize (gptq=False: размер GPTQ-файла тот же, гейт
test_quantize_gptq). Пресеты: red -- REDUCTION; comp -- COMPRESSION без autopick; wprop -- COMPRESSION + autopick +0.5%
по measure-JSON. Файлы -- ~/rwkvq/files_3009/<метка>_<плечо>.rwkvq (оставляются; удаляет владелец).
    python matrix_files_3009.py <ckpt> <метка> <плечи> [measure.json]   -> строки 'РАЗМЕР метка плечо байты'"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant import api
CK, LAB, ARMS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
MJ = sys.argv[4] if len(sys.argv) > 4 else None
TOK = os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
D = os.path.expanduser("~/rwkvq/files_3009"); os.makedirs(D, exist_ok=True)
for a in ARMS:
    t0 = time.time(); p = os.path.join(D, "%s_%s.rwkvq" % (LAB, a))
    kw = dict(preset="reduction") if a == "red" else dict(preset="compression", autopick=(a == "wprop"), measure=MJ or "auto")
    api.quantize(CK, p, tokenizer=TOK, gptq=False, verbose=False, **kw)
    print("РАЗМЕР %s %s %d байт, %.0f с" % (LAB, a, os.path.getsize(p), time.time() - t0), flush=True)
print("ckpt %s %d байт" % (os.path.basename(CK), os.path.getsize(CK)), flush=True)
