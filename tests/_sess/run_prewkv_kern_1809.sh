#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
L=/Users/s/Develop/WKV-kvant/decode_prewkv_kern_1809.log
cd /Users/s/Develop/rwkv-quant
: > $L
for f in compression_0p1b.rwkvq compression_0p4b.rwkvq compression_0p1b.rwkvq; do
  echo "##### $f $(date +%T)" >> $L
  RWKVQ_COMP=$f RWKVQ_ROUNDS=15 RWKVQ_ARMS=full,prewkv_id,prewkv_kern $PY tests/_sess/bench_decode_rest_ab.py >> $L 2>&1
  sleep 15
done
echo "##### DONE $(date +%T)" >> $L
