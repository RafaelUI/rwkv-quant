"""03.10, СЕРВЕР: файлы .rwkvq через quantize() для статьи -- плечи абляции и реплики GPTQ.
Плечо задаётся окружением: RWKVQ_AUTOPICK / RWKVQ_GPTQ = 0 | 1 | пусто (умолчание пресета);
RWKVQ_PERM = сид перестановки ПОРЯДКА 600 окон калибровки GPTQ (пусто -- порядок корпуса пакета). Перестановка меняет
только порядок суммирования H (те же окна) -- реплика для оценки разброса GPTQ между численно эквивалентными прогонами.
    python make_files_0310.py <ckpt> <метка> <reduction|compression> <тег>   -> ~/rwkvq/files_0310/<метка>_<тег>.rwkvq"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant import api
from rwkv_quant.calibration import gptq as G
CK, LAB, PR, TAG = sys.argv[1:5]
TOK = os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
flag = lambda n: {"": None, "0": False, "1": True}[os.environ.get(n, "")]
AP, GP, PERM = flag("RWKVQ_AUTOPICK"), flag("RWKVQ_GPTQ"), os.environ.get("RWKVQ_PERM", "")
D = os.path.expanduser("~/rwkvq/files_0310"); os.makedirs(D, exist_ok=True)
p = os.path.join(D, "%s_%s.rwkvq" % (LAB, TAG))
calib = None
if PERM != "":
    tok, label = G.calib_tokens(TOK, None, G.N_WINDOWS)
    perm = torch.randperm(tok.shape[0], generator=torch.Generator().manual_seed(int(PERM)))
    assert sorted(perm.tolist()) == list(range(tok.shape[0])) and perm.tolist() != list(range(tok.shape[0]))
    calib = tok[perm].contiguous()
    print("калибровка: %s, перестановка сид %s, первые окна %s" % (label, PERM, perm[:8].tolist()), flush=True)
print("старт %s %s %s тег %s autopick %s gptq %s perm %r dev %s" % (time.ctime(), LAB, PR, TAG, AP, GP, PERM, os.environ.get("RWKVQ_DEVICE")), flush=True)
t0 = time.time()
api.quantize(CK, p, preset=PR, tokenizer=TOK, device=os.environ.get("RWKVQ_DEVICE"), verbose=True, autopick=AP, gptq=GP, gptq_calib=calib)
print("ФАЙЛ %s %s %d байт, %.0f с" % (LAB, TAG, os.path.getsize(p), time.time() - t0), flush=True)
