#!/bin/sh
# 19.09: ppl НАПРЯМУЮ С ЗАПИСАННОГО ФАЙЛА (долг 2 от 09.09) для строк COMPRESSION
# 0.1B и 2.9B; 0.4B и 1.5B сняты в run_ce_calib_1909.sh (плечо repo).
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/ppl_from_file_1909.log
cd /Users/s/Develop/rwkv-quant
: > $L
R01=$KV/kl_ref_rwkv7-g1d-0p1b_eval_corpus_multiling_38x512_fp32.npy
R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
one() { echo "##### $1 $2 $(date +%T) своп $(sysctl -n vm.swapusage | awk '{print $6}')" >> $L; RWKVQ_KL_REF=$3 $PY tests/_sess/ce_real_path_1909.py $4 $2 $KV/ppl_from_file_$1_1909.json >> $L 2>&1; }
one 0p1b ref $R01 --ref
one 0p1b compression $R01 $KV/compression_0p1b.rwkvq
one 2p9b ref $R29 --ref
one 2p9b compression $R29 $KV/compression_2p9b.rwkvq
for s in 0p1b 2p9b; do echo "##### отчёт $s" >> $L; $PY tests/_sess/ce_real_path_1909.py --report $KV/ppl_from_file_${s}_1909.json >> $L 2>&1; done
echo "##### КОНЕЦ $(date +%T) своп $(sysctl -n vm.swapusage | awk '{print $6}')" >> $L
