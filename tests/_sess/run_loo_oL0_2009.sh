#!/bin/sh
# 20.09: o_proj слоя 0 в bf16 -- 0.4B (решающая точка правила) и REDUCTION на трёх масштабах.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/loo_oL0_2009.log
S=$HOME/.cache/rwkv-quant/act_stats
cd /Users/s/Develop/rwkv-quant
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
arm() {  # пресет масштаб чекпоинт статистика эталон плечо json
  echo "##### $1 $2 $6 $(date +%T) своп $(sw)" >> $L
  RWKVQ_PRESET=$1 RWKVQ_CKPT=$3 RWKVQ_ACT_STATS=$4 RWKVQ_KL_REF=$5 RWKVQ_OUT=$7 \
    /usr/bin/time -l $PY tests/_sess/loo_compression_kl_1909.py $6 2>&1 | grep -E "KL|Error|Traceback|SystemExit|peak memory|задет" >> $L
}
: > $L
C04=/Users/s/rwkv7-g1d-0.4b-ctx8192.pth; R04=$KV/kl_ref_rwkv7-g1d-0p4b-ctx8192_eval_corpus_multiling_38x512_fp32.npy; A04=$S/act_cb9522cbd3ea0e4c.pt
C15=$KV/rwkv7-g1h-1.5b-ctx10240.pth; R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy; A15=$S/act_12dbb2d3ac61eb20.pt
C29=/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth; R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy; A29=$S/act_cdc6e20e40120763.pt
for a in base oL0; do arm compression 0p4b $C04 $A04 $R04 $a $KV/loo_compression_0p4b_1909.json; done
arm compression 1p5b $C15 $A15 $R15 oL0 $KV/loo_compression_1p5b_1909.json
for a in base oL0; do arm reduction 0p4b $C04 $A04 $R04 $a $KV/loo_reduction_0p4b_2009.json; done
for a in base oL0; do arm reduction 1p5b $C15 $A15 $R15 $a $KV/loo_reduction_1p5b_2009.json; done
for a in base oL0; do arm reduction 2p9b $C29 $A29 $R29 $a $KV/loo_reduction_2p9b_2009.json; done
echo "##### КОНЕЦ $(date +%T) своп $(sw)" >> $L
