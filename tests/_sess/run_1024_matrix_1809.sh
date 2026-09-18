#!/bin/sh
# 18.09: pp1024/tg1024, наши против llama.cpp, ЧЕРЕДУЯ плечи и повторяя пары
# 1.5B/2.9B вторым проходом (дрейф). Машина холодная, на зарядке; caffeinate
# держит её от засыпания -- две прошлые цепочки погибли именно во сне.
# В конце -- перемер ppl 1.5B тем же прибором и той же честной калибровкой,
# что 0.1B/0.4B/2.9B, чтобы все четыре строки README были сняты одинаково.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
A=$KV/artifacts
LB=/opt/homebrew/bin/llama-bench
cd /Users/s/Develop/rwkv-quant
ours() { $PY tests/_sess/bench_scales_1024.py $1 5 2>&1 | grep -vE "emb-gather"; }
llama() { echo "--- llama $1 $(date +%T)"; $LB -m $KV/$1.gguf -p 1024 -n 1024 -ngl 99 -r 3 -b 1024 -ub 1024 -o md 2>&1 | grep -E "^[|] rwkv"; }
pass() {
  echo "===== ПРОХОД $1 $(date +%T) батарея $(pmset -g batt | grep -o "[0-9]*%" | head -1)"
  ours $KV/compression_v2_cand.rwkvq
  llama rwkv7-1p5b-Q4_K_M
  ours $KV/reduction_1p5b_0709.rwkvq
  llama rwkv7-1p5b-Q6_K
  ours $KV/compression_2p9b.rwkvq
  llama rwkv7-g1h-2.9b-20260710-ctx10240-Q4_K_M
  sysctl -n vm.swapusage
}
pass A
pass B
echo "===== МАЛЫЕ $(date +%T)"
ours $KV/compression_0p1b.rwkvq
ours $A/reduction_0p1b_1709.rwkvq
llama rwkv7-g1d-0.1b-20260129-ctx8192-q8_0
ours $KV/compression_0p4b.rwkvq
ours $A/reduction_0p4b_1709.rwkvq
llama rwkv7-g1d-0.4b-20260210-ctx8192-q8_0
echo "===== ppl 1.5B честная калибровка $(date +%T)"
RWKVQ_CKPT=$KV/rwkv7-g1h-1.5b-ctx10240.pth RWKVQ_EVAL_JSON=$KV/eval_multiling_1p5b_1809.json RWKVQ_CONFIGS="bf16 baseline,COMPRESSION" $PY tests/eval_multiling.py $KV/act_stats_calibml.pt 2>&1 | grep -vE "^    ppl |^chunk"  | tail -12
echo "[$(date +%T)] КОНЕЦ_МАТРИЦЫ своп $(sysctl -n vm.swapusage | cut -d= -f3 | cut -d" " -f2)"
