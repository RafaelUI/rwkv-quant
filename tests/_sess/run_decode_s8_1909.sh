#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
L=/Users/s/Develop/WKV-kvant/decode_s8_1909.log
cd /Users/s/Develop/rwkv-quant
: > $L
for f in compression_v2_cand.rwkvq compression_0p1b.rwkvq compression_v2_cand.rwkvq; do
  echo "##### $f $(date +%T)" >> $L
  RWKVQ_COMP=$f RWKVQ_ROUNDS=11 RWKVQ_ARMS=full,all RWKVQ_S8=full,all $PY tests/_sess/bench_decode_rest_ab.py >> $L 2>&1
  sleep 15
done
echo "##### DONE $(date +%T)" >> $L
