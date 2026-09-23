#!/bin/bash
# 22.09 ночь: лестница ВНИЗ (weakspot down) после добора лестниц вверх.
cd ~/rwkvq; W=~/rwkvq/rwkv-quant/tests/_sess/weakspot_2109.py; PY=~/venv/bin/python
export OMP_NUM_THREADS=4
log() { echo "$(date +%T) $*" >> ~/rwkvq/down_2209.log; }
log "жду run_ladder_full_2209"
while pgrep -f "[r]un_ladder_full_2209" > /dev/null; do sleep 60; done
one() { m=$1; devs=$2
  ( source env_$m.sh; export RWKVQ_DEVICE=${devs%%,*} RWKVQ_DEVICES=$devs
    log "старт $m на $devs"
    $PY $W down ws_${m}_down.json > ws_${m}_down.log 2>&1
    log "конец $m код $?" ); }
q0() { one 1p5b cuda:0; one 0p1b cuda:0; one 0p4b cuda:0; }
q1() { one 2p9b cuda:1; }
q23() { one 7p2b cuda:2,cuda:3; }
log "СТАРТ"
q0 & q1 & q23 & wait
log "ВСЁ"
