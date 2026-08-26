# -*- coding: utf-8 -*-
"""ПЕРЕСБОРКА 2.9B НЫНЕШНИМ ПРЕСЕТОМ (приоритет 1).

Прежний файл reduction_sym_head8_2p9b.rwkvq -- пресет 13.08: proj и emb на
ШЕСТИ битах, emb вообще в sb6 (2484.3 МБ). Нынешний REDUCTION требует
proj/emb/head @8 в Q6_K-раскладке; ожидание по таблице пресета -- 2736.8 МБ.

КАЛИБРОВКА -- ТОЛЬКО СВОЯ (закон 15). Берётся /tmp/act_stats_2p9b_ml.pt
(193 тензора, каналы 2560/10240 -- проверено), а не "auto": так рецепт
совпадает с тем, которым собран эталон 1.5B 16.08, и статистика не снята
с текста, на котором меряется ppl. Валидация "auto" на 2.9B -- отдельная
работа, здесь менялся бы второй фактор сразу (закон 5).
"""
import os, sys, time, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CKPT  = "/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth"
OUT   = "/tmp/reduction_2p9b_new.rwkvq"
STATS = "/tmp/act_stats_2p9b_ml.pt"
TOK   = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"

def swap():
    return os.popen("sysctl -n vm.swapusage").read().strip()

def main():
    import torch
    import rwkv_quant
    print("rwkv_quant из:", rwkv_quant.__file__, flush=True)

    # Принадлежность статистики -- ГРОМКО, до долгого прогона.
    st = torch.load(STATS, map_location="cpu")
    ch = sorted({int(v.numel()) for v in st.values()})
    print("act_stats: %d тензоров, каналы %s" % (len(st), ch), flush=True)
    assert ch == [2560, 10240], "статистика не от 2.9B: %s" % ch
    del st

    print("своп ДО:", swap(), flush=True)
    t0 = time.time()
    rwkv_quant.quantize(CKPT, OUT, preset="reduction", real_gw=True,
                        verbose=True, tokenizer=TOK, act_stats=STATS)
    dt = time.time() - t0
    print("своп ПОСЛЕ:", swap(), flush=True)

    sz = os.path.getsize(OUT) / 2**20
    h = hashlib.md5()
    with open(OUT, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    print("=== ГОТОВО: %.1f с, %.1f МБ, md5 %s" % (dt, sz, h.hexdigest()), flush=True)
    print("=== ожидание по таблице пресета: 2736.8 МБ, было (13.08): 2484.3 МБ", flush=True)

main()
