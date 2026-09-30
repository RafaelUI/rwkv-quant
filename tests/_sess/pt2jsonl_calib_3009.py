"""30.09: калибровка GPTQ v2 (gptq_calib2_2809.pt, 600 окон x 512 токенов world) -> текстовый JSONL пакета
(по окну на строку: text, lang, src), чтобы библиотека разбирала его словарём модели. Проверка: каждое окно,
декодированное и снова закодированное WorldTokenizer, даёт РОВНО те же 512 токенов (иначе пакет калибровал бы
не тем, что мерилось 28-30.09). Окна с невалидным UTF-8 на краях -- в отчёт.
    python pt2jsonl_calib_3009.py <out.jsonl>"""
import ast, collections, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant.calibration import act_stats as A
V = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
d = torch.load(os.path.expanduser("~/Develop/WKV-kvant/gptq_calib2_2809.pt"), weights_only=False)
idx2b = {}
for line in open(V, encoding="utf-8"):
    i = int(line[:line.index(" ")]); r = line[line.index(" "):line.rindex(" ")].strip()
    x = ast.literal_eval(r); idx2b[i] = x.encode("utf-8") if isinstance(x, str) else x
enc = A._encoder(V)
ok, bad_utf, bad_rt = 0, [], []
rows = []
for j, w in enumerate(d["tokens"].tolist()):
    b = b"".join(idx2b[t] for t in w)
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        bad_utf.append(j); continue
    if enc(s) != w:
        bad_rt.append(j); continue
    ok += 1
    rows.append(dict(text=s, lang=d["lang"][j], src=d["src"][j]))
print("окон %d: туда-обратно побитно %d; невалидный UTF-8 %d %s; иные токены %d %s" % (
    len(d["lang"]), ok, len(bad_utf), bad_utf[:10], len(bad_rt), bad_rt[:10]))
print("выпавшие по источникам:", collections.Counter(d["src"][j] for j in bad_utf + bad_rt))
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("->", sys.argv[1], os.path.getsize(sys.argv[1]), "байт")
