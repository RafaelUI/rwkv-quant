#!/bin/sh
# 20.09: пересборка эталонов KL 1.5B/2.9B по одному + досчёт LOO + плечо o5.
# Эталон принимается ЧИСЛОМ base-плеча (закон: не размером), иначе стоп.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/loo_2009.log
S=$HOME/.cache/rwkv-quant/act_stats
cd /Users/s/Develop/rwkv-quant
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
ref() {  # чекпоинт эталон
  echo "##### ЭТАЛОН $2 $(date +%T) своп $(sw)" >> $L
  RWKVQ_CKPT=$1 RWKVQ_KL_NSEQ=38 RWKVQ_KL_REF=$2 /usr/bin/time -l $PY tests/ablate_subgroups.py --ref 2>&1 \
    | grep -E "эталон|38/38|Error|Traceback|peak memory|своп" >> $L
  echo "##### ЭТАЛОН ГОТОВ $(date +%T) своп $(sw) размер $(stat -f %z $2)" >> $L
}
arm() {  # масштаб чекпоинт статистика эталон плечо
  echo "##### $1 $5 $(date +%T) своп $(sw)" >> $L
  RWKVQ_CKPT=$2 RWKVQ_ACT_STATS=$3 RWKVQ_KL_REF=$4 RWKVQ_OUT=$KV/loo_compression_$1_1909.json \
    /usr/bin/time -l $PY tests/_sess/loo_compression_kl_1909.py $5 2>&1 | grep -E "KL|Error|Traceback|SystemExit|peak memory|задет" >> $L
}
check() {  # масштаб ожидаемый_KL
  got=$(grep "^base: KL" $L | tail -1 | awk '{print $3}')
  if [ "$got" != "$2" ]; then echo "##### СТОП: $1 base KL $got != $2 -- эталон не принят" >> $L; exit 1; fi
  echo "##### ЭТАЛОН $1 ПРИНЯТ: base KL $got" >> $L
}
: > $L
C15=$KV/rwkv7-g1h-1.5b-ctx10240.pth; R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
C29=/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth; R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
ref $C15 $R15
arm 1p5b $C15 $S/act_12dbb2d3ac61eb20.pt $R15 base; check 1p5b 0.035992
for a in ffn_k o5; do arm 1p5b $C15 $S/act_12dbb2d3ac61eb20.pt $R15 $a; done
ref $C29 $R29
arm 2p9b $C29 $S/act_cdc6e20e40120763.pt $R29 base; check 2p9b 0.052048
for a in ffn_v head emb lora o5; do arm 2p9b $C29 $S/act_cdc6e20e40120763.pt $R29 $a; done
echo "##### КОНЕЦ $(date +%T) своп $(sw)" >> $L
