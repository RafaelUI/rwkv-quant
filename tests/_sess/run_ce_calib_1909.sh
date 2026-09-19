#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant; A=$KV/artifacts
L=$KV/ce_calib_1909.log
cd /Users/s/Develop/rwkv-quant
: > $L
R04=$KV/kl_ref_rwkv7-g1d-0p4b-ctx8192_eval_corpus_multiling_38x512_fp32.npy
R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
one() { echo "##### $1 $2 $(date +%T) своп $(sysctl -n vm.swapusage | awk '{print $6}')" >> $L; RWKVQ_KL_REF=$3 $PY tests/_sess/ce_real_path_1909.py $4 $2 $KV/ce_calib_$1_1909.json >> $L 2>&1; }
one 0p4b ref $R04 --ref
one 0p4b repo $R04 $KV/compression_0p4b.rwkvq
one 0p4b reposmall $R04 $A/compression_0p4b_reposmall_1909.rwkvq
one 0p4b narrow $R04 $A/compression_0p4b_narrow_1909.rwkvq
one 1p5b ref $R15 --ref
one 1p5b repo $R15 $KV/compression_v2_cand.rwkvq
one 1p5b reposmall $R15 $A/compression_1p5b_reposmall_1909.rwkvq
one 1p5b narrow $R15 $A/compression_1p5b_narrow_1909.rwkvq
for s in 0p4b 1p5b; do echo "##### отчёт $s" >> $L; $PY tests/_sess/ce_real_path_1909.py --report $KV/ce_calib_${s}_1909.json >> $L 2>&1; done
echo "##### КОНЕЦ $(date +%T) своп $(sysctl -n vm.swapusage | awk '{print $6}')" >> $L
