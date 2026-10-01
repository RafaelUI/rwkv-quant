"""01.10, СЕРВЕР: файлы .rwkvq С УМОЛЧАНИЯМИ quantize() (REDUCTION: GPTQ; COMPRESSION: autopick + GPTQ) для README --
дальше они качаются на Mac и мерятся реальным путём (Metal). Никаких параметров сверх preset/tokenizer/device: файл
обязан быть тем, что получит пользователь.
    python make_default_files_0110.py <ckpt> <метка> <reduction|compression>   -> ~/rwkvq/files_0110/<метка>_<пресет>.rwkvq"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rwkv_quant import api
CK, LAB, PR = sys.argv[1:4]
D = os.path.expanduser("~/rwkvq/files_0110"); os.makedirs(D, exist_ok=True)
p = os.path.join(D, "%s_%s.rwkvq" % (LAB, PR))
t0 = time.time()
api.quantize(CK, p, preset=PR, tokenizer=os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"),
             device=os.environ.get("RWKVQ_DEVICE"), verbose=True)
print("ФАЙЛ %s %s %d байт, %.0f с" % (LAB, PR, os.path.getsize(p), time.time() - t0), flush=True)
