#!/bin/bash
# 23.09: ПЕРЕСЧЁТ 13.3B -- в run_big_2209 режим down при RWKVQ_DOWN_MIN=4 пропустил 4-битные плечи
# вместе с базой (62 плеча из ~430) -> выбор, граница, joint и отложенные окна 13.3B недействительны.
cd ~/rwkvq; S=~/rwkvq/rwkv-quant/tests/_sess; PY=~/venv/bin/python; W=$S/weakspot_2109.py
export OMP_NUM_THREADS=4 RWKVQ_CORPUS=~/rwkvq/eval_corpus_multiling.pt
log() { echo "$(date +%T) $*" >> ~/rwkvq/redo13_2309.log; }
H=$S/run_heldout_2209.sh; E=/tmp/empty_ovr.json; echo "{}" > $E; D4=cuda:0,cuda:1,cuda:2,cuda:3
mkdir -p invalid_13b_2209 && cp ws_13b_joint2.json ws_13b_ho.json lib_13b_tau*.json sym_13b_tau*.json invalid_13b_2209/ 2>/dev/null
rm -f ws_13b_joint2.json ws_13b_ho.json
log "СТАРТ: down (база всех плеч + шаг вниз)"
( source env_13b.sh; export RWKVQ_DEVICE=cuda:0 RWKVQ_DEVICES=$D4 RWKVQ_DOWN_MIN=4
  $PY $W down ws_13b_down.json > ws_13b_down2.log 2>&1; log "down код $?" )
$PY -c "import json,collections; d=json.load(open('ws_13b_down.json'))['down']; print('плеч', len(d), collections.Counter(e['group'] for e in d.values()))" >> ~/rwkvq/redo13_2309.log
log "lib_select"; $PY $S/lib_select_2209.py > lib_select_13b.log 2>&1
log "граница"; $PY $S/autopick_offline_2209.py > autopick_offline_13b.log 2>&1
( source env_13b.sh; export RWKVQ_DEVICE=cuda:0 RWKVQ_DEVICES=$D4 RWKVQ_LADDER_ARMS=~/rwkvq/cand_13b_bound_tau5.json
  log "добор лестниц: $($PY -c "import json;print(len(json.load(open('cand_13b_bound_tau5.json'))))") кандидатов"
  $PY $W ladder ws_13b_ladder_full.json ws_13b_proxy.json 12 > ws_13b_ladder_full2.log 2>&1; log "лестницы код $?" )
log "полная истина + sym"; $PY $S/autopick_design_2209.py > autopick_design_all.log 2>&1
log "joint"; bash $S/run_joint_full_2209.sh 13b $D4 live:$E lib5:lib_13b_tau5.json lib6:lib_13b_tau6.json lib7:lib_13b_tau7.json sym5:sym_13b_tau5.json sym6:sym_13b_tau6.json sym7:sym_13b_tau7.json
log "отложенные"; bash $H 13b $D4 5 live:$E lib5:lib_13b_tau5.json lib6:lib_13b_tau6.json sym5:sym_13b_tau5.json sym6:sym_13b_tau6.json sym7:sym_13b_tau7.json
log "ВСЁ"
