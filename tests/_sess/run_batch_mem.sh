#!/bin/sh
# ПАМЯТЬ батчевого префилла: один батч на процесс, пик из /usr/bin/time -l.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
S=/Users/s/Develop/rwkv-quant/tests/_sess/bench_prefill_batch_mem.py
for b in 1 2 4 8; do
  echo "===== B=$b ====="
  RWKVQ_B=$b /usr/bin/time -l $PY $S 2>&1 | grep -v "^ *[0-9]* *\(instructions\|cycles\) " 
  echo "--- своп после B=$b ---"
  sysctl -n vm.swapusage
done
echo "ВСЕ БАТЧИ ПРОЙДЕНЫ"
