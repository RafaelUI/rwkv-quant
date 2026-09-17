#!/bin/sh
# Гейт кернеля декванта на всех размерах: ОДНА модель на процесс
# (закон 2), 2.9B последним среди compression -- он самый тяжёлый.
# 17.09: плюс факт включения в рабочий _dequant_w, reduction и мутация.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
G=/Users/s/Develop/rwkv-quant/tests/test_gw_dequant_kernel_parity.py
KV=/Users/s/Develop/WKV-kvant
F="тензоров|xbits|расхожден|рабочий|ГЕЙТ|Error|error|Traceback|peak memory"
for f in compression_0p1b compression_0p4b compression_v2_cand compression_2p9b reduction_1p5b_0709; do
  echo "===== $f ====="
  vm_stat | grep Swapouts
  /usr/bin/time -l $PY $G $KV/$f.rwkvq 2>&1 | grep -E "$F"
  sysctl -n vm.swapusage; vm_stat | grep Swapouts
done
echo "===== МУТАЦИЯ compression_0p1b ====="
MUTATE=1 $PY $G $KV/compression_0p1b.rwkvq 2>&1 | grep -E "$F"
echo "ВСЕ РАЗМЕРЫ ПРОЙДЕНЫ"
