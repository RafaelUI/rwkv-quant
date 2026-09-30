"""30.09: заморозка поведения calibration.gptq.run ДО переделки под API (устройства списком, активации на CPU,
обновление на месте, калибровка извне). 0.1B g1d, пресет COMPRESSION, 16 окон корпуса пакета, damp 0.01.
Хеш каждого поля каждого упакованного тензора + deq. Переделка обязана воспроизвести ПОБИТНО при тех же окнах.
    python gptq_run_freeze_3009.py freeze|check   -> tests/_sess/gptq_run_freeze_3009.json"""
import copy, hashlib, inspect, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import act_stats as A, gptq as G
CK = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
J = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gptq_run_freeze_3009.json")
NW = 16
h = lambda t: hashlib.sha1(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
_, sig = A.collect(CK, TOK)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = os.path.join(A.CACHE_DIR, "act_%s.pt" % sig)
kw = dict(n_windows=NW, damp=0.01, keep_deq=True, verbose=False)
if "calib" in inspect.signature(G.run).parameters:
    enc = A._encoder(TOK)
    chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", open(A.CORPUS, encoding="utf-8").read()) if c.strip()]
    kw["calib"] = torch.tensor(A._windows(chunks, enc, A.SEQ_LEN, NW * A.SEQ_LEN)[:NW], dtype=torch.long)
t0 = time.time()
out, deqs = G.run(CK, TOK, cfg, **kw)
rec = {}
for k in sorted(out):
    rec[k] = {f: h(v) for f, v in sorted(vars(out[k]).items()) if isinstance(v, torch.Tensor)}
    rec[k]["_meta"] = hashlib.sha1(repr(sorted((f, v) for f, v in vars(out[k]).items() if not isinstance(v, torch.Tensor))).encode()).hexdigest()[:16]
    rec[k]["deq"] = h(deqs[k])
    assert torch.isfinite(deqs[k]).all(), k
print("%d тензоров, %.0f с, калибровка %s" % (len(rec), time.time() - t0, "извне" if "calib" in kw else "внутри run"))
if sys.argv[1] == "freeze":
    json.dump(rec, open(J, "w"), indent=0, sort_keys=True); print("заморожено ->", J)
else:
    ref = json.load(open(J))
    assert sorted(ref) == sorted(rec), "набор ключей"
    # 30.09: голова ИСКЛЮЧЕНА из побитной сверки намеренно -- H головы теперь копится батчами по 8 окон (активации
    # на CPU), а не одним матмулом: другой порядок суммирования fp32, и GPTQ на иной H даёт иные коды. Все 71
    # матрица блоков (их H копились батчами и раньше) обязаны совпасть побитно -- это и доказывает, что вход
    # головы тот же; качество головы охраняет test_quantize_gptq (KL файла против RTN).
    bad = [(k, f) for k in ref for f in ref[k] if k != "head.weight" and ref[k][f] != rec[k].get(f)]
    hd = [f for f in ref["head.weight"] if ref["head.weight"][f] != rec["head.weight"].get(f)]
    print("расхождений вне головы: %d из %d полей; голова: %d полей иные (ожидаемо)" % (
        len(bad), sum(len(v) for k, v in ref.items() if k != "head.weight"), len(hd)), bad[:5])
    assert not bad
    print("gptq_run_freeze: блоки ПОБИТНО")
