#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
L=/Users/s/Develop/WKV-kvant/decode_rest_1809.log
cd /Users/s/Develop/rwkv-quant
: > $L
for f in compression_v2_cand.rwkvq compression_0p1b.rwkvq compression_0p4b.rwkvq compression_2p9b.rwkvq compression_v2_cand.rwkvq; do
  echo "##### $f $(date +%T) batt=$(pmset -g batt | grep -o '[0-9]*%')" >> $L
  RWKVQ_COMP=$f RWKVQ_ROUNDS=9 $PY tests/_sess/bench_decode_rest_ab.py >> $L 2>&1
  sleep 20
done
echo "##### DONE $(date +%T)" >> $L
