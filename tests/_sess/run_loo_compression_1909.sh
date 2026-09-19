#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/loo_compression_1909.log
cd /Users/s/Develop/rwkv-quant
: > $L
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
go() {  # масштаб чекпоинт статистика эталон
  for a in base r k v o ffn_k ffn_v head emb lora; do
    echo "##### $1 $a $(date +%T) своп $(sw)" >> $L
    RWKVQ_CKPT=$2 RWKVQ_ACT_STATS=$3 RWKVQ_KL_REF=$4 RWKVQ_OUT=$KV/loo_compression_$1_1909.json \
      /usr/bin/time -l $PY tests/_sess/loo_compression_kl_1909.py $a 2>&1 | grep -E "KL|Error|Traceback|SystemExit|peak memory|задет" >> $L
  done
}
S=$HOME/.cache/rwkv-quant/act_stats
go 1p5b $KV/rwkv7-g1h-1.5b-ctx10240.pth $S/act_12dbb2d3ac61eb20.pt $KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
go 2p9b /Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth $S/act_cdc6e20e40120763.pt $KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
echo "##### КОНЕЦ $(date +%T) своп $(sw)" >> $L
