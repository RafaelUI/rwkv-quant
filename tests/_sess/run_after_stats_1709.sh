#!/bin/sh
# 17.09 ночь: ждёт статистику 2.9B, считает ppl 2.9B (две руки по процессу),
# затем pp1024/tg1024 по всем нашим файлам. Строго по очереди: сбор
# статистики грузит CPU, eval и бенчи -- GPU, смешивать нельзя.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
A=$KV/artifacts
cd /Users/s/Develop/rwkv-quant
i=0
while [ ! -f $A/act_stats_2p9b_calibml.pt ] && [ $i -lt 120 ]; do sleep 15; i=$((i+1)); done
if [ ! -f $A/act_stats_2p9b_calibml.pt ]; then echo "СТАТИСТИКИ 2.9B НЕТ, дальше не иду"; exit 1; fi
sleep 5
echo "[$(date +%T)] статистика 2.9B есть: $(ls -l $A/act_stats_2p9b_calibml.pt | awk "{print \$5}") Б"
for c in bf16 compression; do
  echo "[$(date +%T)] ppl 2.9B $c; своп $(sysctl -n vm.swapusage | cut -d= -f3 | cut -d" " -f2)"
  RWKVQ_ACT_STATS=$A/act_stats_2p9b_calibml.pt RWKVQ_EVAL_JSON=$KV/eval_2p9b_1709.json /usr/bin/time -l $PY tests/eval_2p9b_one.py $c 2>&1 | grep -vE "^    ppl |^chunk" | tail -25
done
echo "===== pp1024 / tg1024 ====="
for f in compression_0p1b compression_0p4b compression_v2_cand artifacts/reduction_1p5b_1709 reduction_1p5b_0709 compression_2p9b artifacts/reduction_2p9b_1709; do
  [ -f $KV/$f.rwkvq ] || { echo "нет $f"; continue; }
  $PY tests/_sess/bench_scales_1024.py $KV/$f.rwkvq 5 2>&1 | grep -vE "emb-gather"
  sysctl -n vm.swapusage
done
echo "[$(date +%T)] КОНЕЦ"
