#!/bin/sh
# 21.09: эмбеддинг -- второй адрес. Плечи поверх нынешнего пресета.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/loo_emb_2109.log
S=$HOME/.cache/rwkv-quant/act_stats
cd /Users/s/Develop/rwkv-quant
: > $L
arm() {
  echo "##### $1 $5 $(date +%T)" >> $L
  RWKVQ_CKPT=$2 RWKVQ_ACT_STATS=$3 RWKVQ_KL_REF=$4 RWKVQ_OUT=$KV/loo_compression_$1_1909.json \
    $PY tests/_sess/loo_compression_kl_1909.py $5 2>&1 | grep -E "KL|Error|Traceback|задет" >> $L
}
C29=/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth; R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
C15=$KV/rwkv7-g1h-1.5b-ctx10240.pth; R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
for a in base e6 e8 emb; do arm 2p9b $C29 $S/act_cdc6e20e40120763.pt $R29 $a; done
for a in base e6; do arm 1p5b $C15 $S/act_12dbb2d3ac61eb20.pt $R15 $a; done
echo "##### КОНЕЦ $(date +%T)" >> $L
