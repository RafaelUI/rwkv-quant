#!/bin/sh
# 21.09: откуда +7-9% декода у файлов правила. 0.4B, по ОДНОЙ правке на файл,
# каждый против базы (compression_nol0, emb 5) ABBA в одном процессе.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python; KV=/Users/s/Develop/WKV-kvant; D=$KV/slow_2109; L=$D/slow.log
cd /Users/s/Develop/rwkv-quant; : > $L
C=/Users/s/rwkv7-g1d-0.4b-ctx8192.pth
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
for v in base oL0_6 oL0_5 oL23_5 vL0_6 rkvL0_6 cvL5_5; do
  $PY tests/_sess/build_rule_2109.py $C $D/$v.json $D/$v.rwkvq 2>&1 | grep -E "готово|Error|Trace" >> $L
done
echo "##### своп в начале $(sw)" >> $L
for v in base oL0_6 oL0_5 oL23_5 vL0_6 rkvL0_6 cvL5_5; do
  echo "##### base(A) vs $v(B) $(date +%T)" >> $L
  $PY tests/_sess/bench_files_ab_2009.py $D/base.rwkvq $D/$v.rwkvq 16 32 2>&1 | grep -E "B-A|Error|Trace" >> $L
done
echo "##### своп в конце $(sw)" >> $L
echo "##### КОНЕЦ $(date +%T)" >> $L
