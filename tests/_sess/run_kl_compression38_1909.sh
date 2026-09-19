#!/bin/sh
# 19.09: KL(fp32 ‖ COMPRESSION) на ВСЕХ 38 окнах (en/ru/sr) -- первые 8 окон
# корпуса оказались целиком русскими, а ppl в README считается по всем 38.
# Эталоны 0.1B/0.4B на 38 окон пересобираются тем же рецептом, что удалённые 17.09.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/kl_compression38_1909.log
cd /Users/s/Develop/rwkv-quant
: > $L
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
ref() {
  echo "##### эталон38 $1 $(date +%T) своп $(sw)" >> $L
  RWKVQ_CKPT=$2 RWKVQ_KL_NSEQ=38 RWKVQ_KL_REF=$3 /usr/bin/time -l $PY tests/ablate_subgroups.py --ref >> $L 2>&1
  echo "  своп после $(sw); файл $(ls -l $3 2>/dev/null | awk '{print $5}') Б" >> $L
}
kl() {
  echo "##### KL38 $1 $4 $(date +%T) своп $(sw)" >> $L
  RWKVQ_CKPT=$5 RWKVQ_KL_NSEQ=38 RWKVQ_KL_REF=$2 RWKVQ_KLREAL_OUT=$KV/kl38_compression_$1_1909.json \
    /usr/bin/time -l $PY tests/_sess/kl_real_path.py $3 $4 >> $L 2>&1
  echo "  своп после $(sw)" >> $L
}
C01=$KV/rwkv7-g1d-0.1b.pth; C04=/Users/s/rwkv7-g1d-0.4b-ctx8192.pth
R01=$KV/kl_ref_rwkv7-g1d-0p1b_eval_corpus_multiling_38x512_fp32.npy
R04=$KV/kl_ref_rwkv7-g1d-0p4b-ctx8192_eval_corpus_multiling_38x512_fp32.npy
R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
ref 0p1b $C01 $R01
ref 0p4b $C04 $R04
kl 0p1b $R01 $KV/compression_0p1b.rwkvq compression
kl 0p4b $R04 $KV/compression_0p4b.rwkvq compression
kl 1p5b $R15 $KV/compression_v2_cand.rwkvq compression
kl 2p9b $R29 $KV/compression_2p9b.rwkvq compression
echo "##### КОНЕЦ $(date +%T)" >> $L
