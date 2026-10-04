#!/bin/sh
# 04.10: cc9 -- наши файлы равного размера с int6 / Q6_K (sym6, sym6h8; RTN и GPTQ), 1.5B и 2.9B, + оценка. НЕ убивать sh цепочки.
cd ~/rwkvq/rwkv-quant-git
export PYTHONPATH=$HOME/rwkvq/rwkv-quant-git:$HOME/rwkvq
PY="$HOME/venv/bin/python -u"; K=$HOME/rwkvq/ckpt; R=$HOME/rwkvq; LOG=$R/chain_eq_0410.log; FD=$R/files_0310; FE=$R/fe0310
C15=$K/rwkv7-g1j-1.5b-20260831-ctx16384.pth; C29=$K/rwkv7-g1j-2.9b-20260831-ctx16384.pth
echo "старт $(date) $(git log --oneline -1 | cut -c1-7)" >> $LOG
# me <карта> <ckpt> <метка> <вид> <gptq 0|1>
me() { T=$4_rtn; [ "$5" = 1 ] && T=$4_gptq
  CUDA_VISIBLE_DEVICES=$1 RWKVQ_DEVICE=cuda RWKVQ_GPTQ=$5 $PY tests/_sess/make_cfg_files_0410.py $2 $3 $4 $T > $FD/$3_$T.log 2>&1; echo "mk $3 $T код $? $(date)" >> $LOG
  CUDA_VISIBLE_DEVICES=$1 RWKVQ_DEVICE=cuda $PY tests/_sess/file_eval_0310.py $FD/$3_$T.rwkvq $2 $FE/$3_$T.json > $FE/$3_$T.log 2>&1; echo "ev $3_$T код $? $(date)" >> $LOG; }
( me 0 $C15 1p5b sym6 0; me 0 $C15 1p5b sym6 1 ) &
( me 1 $C15 1p5b sym6h8 0; me 1 $C15 1p5b sym6h8 1 ) &
( me 2 $C29 2p9b sym6 0; me 2 $C29 2p9b sym6 1 ) &
( me 3 $C29 2p9b sym6h8 0; me 3 $C29 2p9b sym6h8 1 ) &
wait
echo "конец $(date)" >> $LOG
