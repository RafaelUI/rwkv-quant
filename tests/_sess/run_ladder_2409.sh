#!/bin/sh
# 24.09: разложение шага декода COMPRESSION g1j 1.5B и 2.9B (каждый масштаб свой).
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
D=/Users/s/Develop/WKV-kvant/g1j_2309
L=$D/decode_ladder_2409.log
cd /Users/s/Develop/rwkv-quant
: > $L
echo "##### START $(date +%T) своп $(sysctl -n vm.swapusage)" >> $L
for M in 1p5b 2p9b; do
  F=$D/compression_${M}_g1j.rwkvq
  echo "##### depth $M $(date +%T)" >> $L
  RWKVQ_COMP=$F RWKVQ_ARMS=full,L_gemv,L_lora,L_wkvblk,all $PY tests/_sess/probe_decode_depth_2409.py >> $L 2>&1
  for run in 1 2; do
    sleep 20
    echo "##### ladder $M прогон $run $(date +%T)" >> $L
    RWKVQ_COMP=$F RWKVQ_ROUNDS=11 RWKVQ_ARMS=full,gemv0,L_gemv,L_lora,L_wkvblk,L_ln,all $PY tests/_sess/bench_decode_rest_ab_2409.py >> $L 2>&1
  done
  sleep 20
  echo "##### ceiling $M $(date +%T)" >> $L
  RWKVQ_COMP=$F RWKVQ_ROUNDS=11 $PY tests/_sess/bench_gemv_ceiling_2409.py >> $L 2>&1
done
echo "##### DONE $(date +%T) своп $(sysctl -n vm.swapusage)" >> $L
