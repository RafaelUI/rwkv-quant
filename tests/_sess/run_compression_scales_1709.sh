#!/bin/sh
# 17.09: перемер COMPRESSION (новая раскладка с 09.09) на 0.1B/0.4B/2.9B --
# долг 1 / строки README. Статистика ЧЕСТНАЯ: снимается на act_calib_multiling
# (17 окон), ppl меряется на eval_corpus_multiling (38 окон) -- корпуса не
# пересекаются. Один масштаб на процесс (закон 2), 2.9B отдельным прибором.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
A=$KV/artifacts
CAL=$KV/act_calib_multiling.pt
cd /Users/s/Develop/rwkv-quant
log() { echo "[$(date +%T)] $1 $(sysctl -n vm.swapusage | cut -d= -f3 | cut -d" " -f2)"; }
stats() {
  if [ -f $A/act_stats_$1_calibml.pt ]; then echo "статистика $1 уже есть"; return; fi
  log "сбор статистики $1"
  RWKVQ_CKPT=$2 /usr/bin/time -l $PY tests/collect_act_stats.py $CAL $A/act_stats_$1_calibml.pt ":" 2>&1 | grep -E "saved|chunk 17|peak memory|Error"
}
stats 0p1b $KV/rwkv7-g1d-0.1b.pth
log "ppl 0.1B"
RWKVQ_CKPT=$KV/rwkv7-g1d-0.1b.pth RWKVQ_EVAL_JSON=$KV/eval_multiling_0p1b_1709.json RWKVQ_CONFIGS="bf16 baseline,COMPRESSION" /usr/bin/time -l $PY tests/eval_multiling.py $A/act_stats_0p1b_calibml.pt 2>&1 | grep -vE "^chunk|^  окно"
stats 0p4b /Users/s/rwkv7-g1d-0.4b-ctx8192.pth
log "ppl 0.4B"
RWKVQ_CKPT=/Users/s/rwkv7-g1d-0.4b-ctx8192.pth RWKVQ_EVAL_JSON=$KV/eval_multiling_0p4b_1709.json RWKVQ_CONFIGS="bf16 baseline,COMPRESSION" /usr/bin/time -l $PY tests/eval_multiling.py $A/act_stats_0p4b_calibml.pt 2>&1 | grep -vE "^chunk|^  окно"
stats 2p9b /Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth
for c in bf16 compression; do
  log "ppl 2.9B $c"
  RWKVQ_ACT_STATS=$A/act_stats_2p9b_calibml.pt RWKVQ_EVAL_JSON=$KV/eval_2p9b_1709.json /usr/bin/time -l $PY tests/eval_2p9b_one.py $c 2>&1 | grep -vE "^chunk|^  окно"
done
log КОНЕЦ
