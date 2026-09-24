#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
D=/Users/s/Develop/WKV-kvant/g1j_2309
L=$D/generate_2409.log
cd /Users/s/Develop/rwkv-quant
: > $L
for M in 1p5b 2p9b 1p5b 2p9b; do
  sleep 20
  echo "##### $M $(date +%T)" >> $L
  RWKVQ_COMP=$D/compression_${M}_g1j.rwkvq $PY tests/_sess/bench_generate_2409.py 16 32 >> $L 2>&1
done
echo "##### DONE $(date +%T)" >> $L
