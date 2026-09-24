#!/bin/sh
# 24.09: та же лестница, что run_ladder_2409, в ритме конвейера async_eval.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
D=/Users/s/Develop/WKV-kvant/g1j_2309
L=$D/decode_ladder_async_2409.log
cd /Users/s/Develop/rwkv-quant
: > $L
for M in 1p5b 2p9b; do
  for run in 1 2; do
    sleep 20
    echo "##### async ladder $M прогон $run $(date +%T)" >> $L
    RWKVQ_ASYNC=1 RWKVQ_COMP=$D/compression_${M}_g1j.rwkvq RWKVQ_ROUNDS=11 RWKVQ_ARMS=full,gemv0,L_gemv,L_lora,L_wkvblk,L_ln,all $PY tests/_sess/bench_decode_rest_ab_2409.py >> $L 2>&1
  done
done
echo "##### DONE $(date +%T)" >> $L
