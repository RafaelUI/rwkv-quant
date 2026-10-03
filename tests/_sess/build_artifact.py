# -*- coding: utf-8 -*-
"""ПЕРЕСБОРКА ПОТЕРЯННЫХ АРТЕФАКТОВ (17.09): build_artifact.py <рецепт> <масштаб>.

Рецепты:
  reduction  -- НЫНЕШНИЙ пресет REDUCTION (рецепт build_2p9b_preset.py:
                real_gw, своя статистика _ml, не auto). Замена
                reduction_*_preset16aug; имя НЕ прежнее (закон 30).
  sym_head8  -- ablate_sym_composite.CONFIGS["reduction_sym_head8"]: база
                LEGACY_REDUCTION, sym на cmix/proj/head, head@8.
  legacy     -- LEGACY_REDUCTION (int6 asym, есть gw_qh2) -- замена
                /tmp/reduction_v2 для test_gw_nb_parity. В отличие от
                сборки 15.08 -- СО статистикой (та выродила AW).
Файл пишется в artifacts с суффиксом _1709; существующий не затирается.
"""
import os, sys, time, hashlib
R = "/Users/s/Develop/rwkv-quant"
sys.path.insert(0, R)
sys.path.insert(0, R + "/tests")
A = "/Users/s/Develop/WKV-kvant/artifacts"
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
CK = {"1p5b": ("/Users/s/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth",
               A + "/act_stats_1p5b_ml.pt", [2048, 8192]),
      # 17.09 ночь: 0.1B и 0.4B добавлены под таблицу скорости (нужна пара к
      # Q8_0); статистика -- ЧЕСТНАЯ calibml, снятая в тот же день под ppl.
      "0p1b": ("/Users/s/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth",
               A + "/act_stats_0p1b_calibml.pt", None),
      "0p4b": ("/Users/s/rwkv7-g1d-0.4b-ctx8192.pth",
               A + "/act_stats_0p4b_calibml.pt", None),
      "2p9b": ("/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth",
               A + "/act_stats_2p9b_ml.pt", [2560, 10240])}
recipe, scale = sys.argv[1], sys.argv[2]
ckpt, stats, chans = CK[scale]
os.environ["RWKVQ_ACT_STATS"] = stats
# 03.10: суффикс -- из RWKVQ_ART_SUFFIX (артефакты _1709 удалены уборкой 29.09; пересборка нынешним кодом
# получает НОВОЕ имя, закон 30). GPTQ при пересборке НЕ включается: рецепты -- RTN, как 17.09.
SUF = os.environ.get("RWKVQ_ART_SUFFIX", "_1709")
out = "%s/%s_%s%s.rwkvq" % (A, recipe, scale, SUF)
assert not os.path.exists(out), "уже есть: " + out
import torch
import rwkv_quant
print("rwkv_quant из:", rwkv_quant.__file__, flush=True)
st = torch.load(stats, map_location="cpu")
ch = sorted({int(v.numel()) for v in st.values()})
print("act_stats: %d тензоров, каналы %s" % (len(st), ch), flush=True)
assert chans is None or ch == chans, "статистика не от этого масштаба: %s" % ch
del st
config = None
if recipe != "reduction":
    import ablate_sym_composite as asc
    config = asc.base_cfg() if recipe == "legacy" else asc.CONFIGS["reduction_sym_head8"]()
    assert config.act_stats_path == stats, config.act_stats_path
print("рецепт", recipe, "config", config, flush=True)
sw = os.popen("sysctl -n vm.swapusage").read().strip()
t0 = time.time()
rwkv_quant.quantize(ckpt, out, preset="reduction", config=config, real_gw=True,
                    verbose=True, tokenizer=TOK, act_stats=stats, gptq=False)
dt = time.time() - t0
h = hashlib.md5()
with open(out, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
print("своп до:", sw, flush=True)
print("своп после:", os.popen("sysctl -n vm.swapusage").read().strip(), flush=True)
print("=== ГОТОВО %s: %.1f с, %d Б, md5 %s" % (out, dt, os.path.getsize(out), h.hexdigest()), flush=True)
