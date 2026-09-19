"""Статистика AW на РЕПОЗИТОРНОМ корпусе, урезанном до объёма узкого (19.09).
Цель -- развести объём и домен: полный репозиторный = 23 окна x 511 = 11753
токена, узкий act_calib_multiling = 17 x 511 = 8687. Урезание -- НЕ выкидыванием
окон (это меняет состав: последние чанки -- код и китайский), а укорочением
каждого из тех же 23 окон до 379 (23 x 378 = 8694 токена): состав по чанкам тот
же, объём -- как у узкого. Оговорка: позиции в окне только 0..377.
    python collect_reposmall_1909.py <чекпоинт> <выход.pt>
"""
import sys, re, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import torch
import rwkv_quant.models.rwkv7_ref as ref_mod
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
from rwkv_quant.calibration import act_stats as A

TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
L = 379
ck, out = sys.argv[1], sys.argv[2]
enc = A._encoder(TOK)
text = open(A.CORPUS, encoding="utf-8").read()
chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", text) if c.strip()]
wins = A._windows(chunks, enc, A.SEQ_LEN, A.TOKEN_BUDGET)
print("полный: %d окон x %d" % (len(wins), A.SEQ_LEN))
data = torch.tensor([w[:L] for w in wins], dtype=torch.int32)
print("урезанный: %d окон x %d -> %d токенов в прямом проходе" % (data.shape[0], L, data.shape[0] * (L - 1)))
model = RWKV7Ref(ck, device="cpu", dtype=torch.bfloat16)
ref_mod.ACT_RECORDER = {}
t0 = time.time()
with torch.no_grad():
    for i in range(data.shape[0]):
        model.forward(data[i:i + 1, :-1])
stats = {k: (ss / max(n, 1)) for k, (ss, n) in ref_mod.ACT_RECORDER.items()}
ref_mod.ACT_RECORDER = None
torch.save(stats, out)
print("снято %d тензоров за %.0f с -> %s" % (len(stats), time.time() - t0, out))
