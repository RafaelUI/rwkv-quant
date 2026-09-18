#!/bin/sh
# 17.09 ночь: ждёт конца предыдущей цепочки, собирает reduction для 0.1B/0.4B
# (пара по размеру к Q8_0), снимает их pp1024/tg1024, затем ту же пару чисел
# у llama.cpp по всем пяти gguf. Наши и их прогоны НЕ пересекаются во времени.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
A=$KV/artifacts
LB=/opt/homebrew/bin/llama-bench
L=$KV/scales_1024_1709.log
cd /Users/s/Develop/rwkv-quant
i=0
while ! grep -q "КОНЕЦ" $L && [ $i -lt 160 ]; do sleep 15; i=$((i+1)); done
grep -q "КОНЕЦ" $L || { echo "предыдущая цепочка не закончилась"; exit 1; }
sleep 5
for s in 0p1b 0p4b; do
  echo "[$(date +%T)] сборка reduction $s"
  $PY tests/_sess/build_artifact.py reduction $s 2>&1 | grep -E "ГОТОВО|Error|Traceback|уже есть"
done
for s in 0p1b 0p4b; do
  [ -f $A/reduction_${s}_1709.rwkvq ] && $PY tests/_sess/bench_scales_1024.py $A/reduction_${s}_1709.rwkvq 5 2>&1 | grep -vE "emb-gather"
done
echo "===== LLAMA pp1024 / tg1024 ====="
for g in rwkv7-g1d-0.1b-20260129-ctx8192-q8_0 rwkv7-g1d-0.4b-20260210-ctx8192-q8_0 rwkv7-1p5b-Q4_K_M rwkv7-1p5b-Q6_K rwkv7-g1h-2.9b-20260710-ctx10240-Q4_K_M; do
  echo "--- $g $(date +%T) батарея $(pmset -g batt | grep -o "[0-9]*%" | head -1)"
  $LB -m $KV/$g.gguf -p 1024 -n 1024 -ngl 99 -r 3 -b 1024 -ub 1024 -o md 2>&1 | grep -E "^[|] rwkv|build"
  sysctl -n vm.swapusage
done
echo "[$(date +%T)] КОНЕЦ_LLAMA"
