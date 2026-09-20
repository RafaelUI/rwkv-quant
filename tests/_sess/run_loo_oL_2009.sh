#!/bin/sh
# 20.09: послойный LOO o_proj -- слои с массивными каналами входа.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/loo_oL_2009.log
S=$HOME/.cache/rwkv-quant/act_stats
cd /Users/s/Develop/rwkv-quant
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
arm() {
  echo "##### $1 $5 $(date +%T) своп $(sw)" >> $L
  RWKVQ_CKPT=$2 RWKVQ_ACT_STATS=$3 RWKVQ_KL_REF=$4 RWKVQ_OUT=$KV/loo_compression_$1_1909.json \
    /usr/bin/time -l $PY tests/_sess/loo_compression_kl_1909.py $5 2>&1 | grep -E "KL|Error|Traceback|SystemExit|peak memory|задет" >> $L
}
: > $L
C15=$KV/rwkv7-g1h-1.5b-ctx10240.pth; R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
C29=/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth; R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
for a in oL0 oL31 oL0+31; do arm 2p9b $C29 $S/act_cdc6e20e40120763.pt $R29 $a; done
for a in oL23 oL0+23; do arm 1p5b $C15 $S/act_12dbb2d3ac61eb20.pt $R15 $a; done
echo "##### КОНЕЦ $(date +%T) своп $(sw)" >> $L
