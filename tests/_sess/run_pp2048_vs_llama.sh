#!/bin/sh
# 17.09: pp2048 наш против llama.cpp (brew llama-bench), пары по размеру:
# compression_v2_cand <-> Q4_K_M, reduction_1p5b_0709 <-> Q6_K. Строго по
# очереди, с повтором первой пары в конце (дрейф безвентиляторной машины).
# У llama два ubatch: 512 (умолчание llama-bench, промпт идёт кусками) и 2048
# (весь промпт одним графом, как у нас).
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
LB=/opt/homebrew/bin/llama-bench
cd /Users/s/Develop/rwkv-quant
ours() { echo "===== НАШ $1 T=2048 $(date +%T) батарея $(pmset -g batt | grep -o "[0-9]*%")"; RWKVQ_COMP=$1 RWKVQ_ARMS=full RWKVQ_T=2048 RWKVQ_ROUNDS=5 $PY tests/_sess/bench_prefill_rest_ab.py 2>&1 | grep -E "full|своп|НЕДЕЙСТВ"; }
llama() { echo "===== LLAMA $1 pp2048 $(date +%T) батарея $(pmset -g batt | grep -o "[0-9]*%")"; $LB -m $KV/$1 -p 2048 -n 0 -ngl 99 -r 5 -b 2048 -ub 512,2048 -o md 2>&1 | grep -E "^[|]|build"; }
echo "llama-bench: $($LB --version 2>&1 | grep -i version | head -1)"
ours compression_v2_cand.rwkvq
llama rwkv7-1p5b-Q4_K_M.gguf
ours reduction_1p5b_0709.rwkvq
llama rwkv7-1p5b-Q6_K.gguf
ours compression_v2_cand.rwkvq
llama rwkv7-1p5b-Q4_K_M.gguf
echo "КОНЕЦ $(date +%T) своп $(sysctl -n vm.swapusage)"
