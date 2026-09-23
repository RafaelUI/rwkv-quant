#!/bin/bash
# 22.09 ночь: добор истинных лестниц для плеч вне прежней лестницы, у которых
# СТРОГАЯ граница e >= 5 (cand_<m>_bound_tau5.json от autopick_offline_2209.py).
# Пишет в копию ws_<m>_ladder.json -> ws_<m>_ladder_full.json (снятое не пересчитывается).
cd ~/rwkvq; W=~/rwkvq/rwkv-quant/tests/_sess/weakspot_2109.py; PY=~/venv/bin/python
export OMP_NUM_THREADS=4
log() { echo "$(date +%T) $*" >> ~/rwkvq/ladder_full_2209.log; }
one() { m=$1; devs=$2
  ( source env_$m.sh; export RWKVQ_DEVICE=${devs%%,*} RWKVQ_DEVICES=$devs RWKVQ_LADDER_ARMS=~/rwkvq/cand_${m}_bound_tau5.json
    [ -f ws_${m}_ladder_full.json ] || cp ws_${m}_ladder.json ws_${m}_ladder_full.json
    log "старт $m на $devs"
    $PY $W ladder ws_${m}_ladder_full.json ws_${m}_proxy.json 12 > ws_${m}_ladder_full.log 2>&1
    log "конец $m код $?" ); }
q0() { one 1p5b cuda:0; one 0p1b cuda:0; one 0p4b cuda:0; }
q1() { one 2p9b cuda:1; }
q23() { one 7p2b cuda:2,cuda:3; }
log "СТАРТ"
q0 & q1 & q23 & wait
log "ВСЁ"
