#!/bin/sh
# Гейт кернеля декванта на всех четырёх размерах: ОДНА модель на процесс
# (закон 2), 2.9B последним -- он самый тяжёлый.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
G=/Users/s/Develop/rwkv-quant/tests/test_gw_dequant_kernel_parity.py
KV=/Users/s/Develop/WKV-kvant
for f in compression_0p1b compression_0p4b compression_v2_cand compression_2p9b; do
  echo "===== $f ====="
  /usr/bin/time -l $PY $G $KV/$f.rwkvq 2>&1 | grep -E "тензоров|xbits|расхожден|ГЕЙТ|Error|error|peak memory"
  sysctl -n vm.swapusage
done
echo "ВСЕ РАЗМЕРЫ ПРОЙДЕНЫ"
