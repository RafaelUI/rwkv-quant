#!/bin/bash
# 22.09 ночь, по замечанию владельца: 7.2B и 13.3B проверяются ТАК ЖЕ, как меньшие модели.
cd ~/rwkvq; S=~/rwkvq/rwkv-quant/tests/_sess; PY=~/venv/bin/python; W=$S/weakspot_2109.py
export OMP_NUM_THREADS=4 RWKVQ_CORPUS=~/rwkvq/eval_corpus_multiling.pt
log() { echo "$(date +%T) $*" >> ~/rwkvq/big_2209.log; }
H=$S/run_heldout_2209.sh; E=/tmp/empty_ovr.json; echo "{}" > $E
log "СТАРТ"
q0() { log "q0: kl_file 2p9b sym6"
  ( cd rwkv-quant; RWKVQ_CKPT=~/rwkvq/ckpt/rwkv7-g1j-2.9b-20260831-ctx16384.pth RWKVQ_KL_REF=~/rwkvq/kl_ref_2p9b_g1j_38x512_fp32.npy \
    $PY tests/_sess/kl_file_2209.py ~/rwkvq/sym_files/2p9b_sym6.rwkvq cuda:0 ~/rwkvq/sym_files/kl_2p9b_sym6.json > ~/rwkvq/sym_files/kl_2p9b_sym6.log 2>&1 )
  log "q0: сверка прибора 1.5B (fake на отложенных vs kl_file)"
  bash $H 1p5b cuda:0 10 live:$E sym6:sym_1p5b_tau6.json sym5:sym_1p5b_tau5.json lib6:lib_1p5b_tau6.json lib5:lib_1p5b_tau5.json; log "q0 конец"; }
q1() { log "q1: 2.9B lib на отложенных"; bash $H 2p9b cuda:1 10 live:$E lib6:lib_2p9b_tau6.json lib5:lib_2p9b_tau5.json lib7:lib_2p9b_tau7.json; log "q1 конец"; }
q23() { log "q23: 7.2B отложенные"
  bash $H 7p2b cuda:2,cuda:3 10 live:$E lib5:lib_7p2b_tau5.json lib6:lib_7p2b_tau6.json lib7:lib_7p2b_tau7.json sym5:sym_7p2b_tau5.json sym6:sym_7p2b_tau6.json sym7:sym_7p2b_tau7.json; log "q23 конец"; }
q0 & q1 & q23 & wait
# ---- 13.3B: весь конвейер на 4 картах
D4=cuda:0,cuda:1,cuda:2,cuda:3
( source env_13b.sh; export RWKVQ_DEVICE=cuda:0 RWKVQ_DEVICES=$D4 RWKVQ_DOWN_MIN=4
  log "13b down (база + шаг вниз, все плечи)"; $PY $W down ws_13b_down.json > ws_13b_down.log 2>&1; log "13b down код $?" )
log "13b lib_select"; $PY $S/lib_select_2209.py > lib_select_13b.log 2>&1
log "13b граница"; $PY $S/autopick_offline_2209.py > autopick_offline_13b.log 2>&1
( source env_13b.sh; export RWKVQ_DEVICE=cuda:0 RWKVQ_DEVICES=$D4 RWKVQ_LADDER_ARMS=~/rwkvq/cand_13b_bound_tau5.json
  [ -f ws_13b_ladder_full.json ] || cp ws_13b_ladder.json ws_13b_ladder_full.json
  log "13b добор лестниц"; $PY $W ladder ws_13b_ladder_full.json ws_13b_proxy.json 12 > ws_13b_ladder_full.log 2>&1; log "13b лестницы код $?" )
log "13b полная истина + sym"; $PY $S/autopick_design_2209.py > autopick_design_all.log 2>&1
log "13b joint (8 окон выбора)"; bash $S/run_joint_full_2209.sh 13b $D4 live:$E lib5:lib_13b_tau5.json lib6:lib_13b_tau6.json lib7:lib_13b_tau7.json sym5:sym_13b_tau5.json sym6:sym_13b_tau6.json sym7:sym_13b_tau7.json
log "13b отложенные"; bash $H 13b $D4 5 live:$E lib5:lib_13b_tau5.json lib6:lib_13b_tau6.json lib7:lib_13b_tau7.json sym5:sym_13b_tau5.json sym6:sym_13b_tau6.json
log "ВСЁ"
