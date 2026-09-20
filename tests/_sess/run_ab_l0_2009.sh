#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
cd /Users/s/Develop/rwkv-quant
L=/Users/s/Develop/WKV-kvant/ab_l0_2p9b_2009.log
: > $L
for pair in "/Users/s/Develop/WKV-kvant/compression_2p9b_nol0_2009.rwkvq /Users/s/Develop/WKV-kvant/compression_2p9b_l0_2009.rwkvq" "/Users/s/Develop/WKV-kvant/compression_2p9b_l0_2009.rwkvq /Users/s/Develop/WKV-kvant/compression_2p9b_nol0_2009.rwkvq" "/Users/s/Develop/WKV-kvant/compression_2p9b_nol0_2009.rwkvq /Users/s/Develop/WKV-kvant/compression_2p9b_nol0_2009.rwkvq"; do
  echo "##### $(date +%T)" >> $L
  /usr/bin/time -l $PY tests/_sess/bench_files_ab_2009.py $pair 16 32 2>&1 | grep -E "мс/ток|своп|peak memory|Error|Traceback" >> $L
done
echo "##### КОНЕЦ $(date +%T)" >> $L
