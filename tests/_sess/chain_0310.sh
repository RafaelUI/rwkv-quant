#!/bin/sh
# 03.10: цепочка для статьи. Фаза 1 -- реплики GPTQ (перестановка порядка окон) и плечо «только GPTQ» 0.1B-2.9B;
# фаза 2 -- оценка всех существующих файлов на трёх наборах; фаза 3 -- 7.2B; фаза 4 -- 13.3B.
# НЕ убивать sh цепочки. Пропуск фазы N: touch ~/rwkvq/skip_0310_N (проверяется перед фазой).
cd ~/rwkvq/rwkv-quant-git
export PYTHONPATH=$HOME/rwkvq/rwkv-quant-git:$HOME/rwkvq
PY="$HOME/venv/bin/python -u"; K=$HOME/rwkvq/ckpt; R=$HOME/rwkvq; LOG=$R/chain_0310.log
MK=tests/_sess/make_files_0310.py; EV=tests/_sess/file_eval_0310.py
FD=$R/files_0310; FE=$R/fe0310; mkdir -p $FD $FE
C01=$K/rwkv7-g1d-0.1b-20260129-ctx8192.pth; C04=$K/rwkv7-g1d-0.4b-20260210-ctx8192.pth
C15=$K/rwkv7-g1j-1.5b-20260831-ctx16384.pth; C29=$K/rwkv7-g1j-2.9b-20260831-ctx16384.pth
C72=$K/rwkv7-g1j-7.2b-20260831-ctx16384.pth; C13=$K/rwkv7-g1j-13.3b-20260831-ctx16384.pth
echo "старт $(date) $(git log --oneline -1 | cut -c1-7)" >> $LOG
# mk <карты> <dev> <ckpt> <метка> <пресет> <тег> <autopick> <gptq> <perm>
mk() { CUDA_VISIBLE_DEVICES=$1 RWKVQ_DEVICE=$2 RWKVQ_AUTOPICK=$7 RWKVQ_GPTQ=$8 RWKVQ_PERM=$9 $PY $MK $3 $4 $5 $6 > $FD/$4_$6.log 2>&1; echo "mk $4 $6 код $? $(date)" >> $LOG; }
# ev <карты> <dev> <файл> <ckpt> <имя>
ev() { CUDA_VISIBLE_DEVICES=$1 RWKVQ_DEVICE=$2 $PY $EV $3 $4 $FE/$5.json > $FE/$5.log 2>&1; echo "ev $5 код $? $(date)" >> $LOG; }
# me = mk + ev
me() { mk "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"; ev "$1" "$2" $FD/$4_$6.rwkvq "$3" $4_$6; }

if [ ! -e $R/skip_0310_1 ]; then
echo "фаза 1 $(date)" >> $LOG
( me 0 cuda $C01 0p1b compression comp_gptq_s0 0 1 ""
  ev 0 cuda $R/files_0110/0p4b_compression.rwkvq $C04 0p4b_comp_both_s0
  for s in 1 2 3 4 5; do me 0 cuda $C04 0p4b compression comp_both_s$s 1 1 $s; done
  me 0 cuda $C04 0p4b compression comp_gptq_s0 0 1 "" ) &
( ev 1 cuda $R/files_0110/1p5b_compression.rwkvq $C15 1p5b_comp_both_s0
  for s in 1 2 3; do me 1 cuda $C15 1p5b compression comp_both_s$s 1 1 $s; done
  me 1 cuda $C15 1p5b compression comp_gptq_s0 0 1 ""
  me 1 cuda $C29 2p9b compression comp_gptq_s0 0 1 "" ) &
( ev 2 cuda $R/files_0110/2p9b_compression.rwkvq $C29 2p9b_comp_both_s0
  for s in 1 2 3; do me 2 cuda $C29 2p9b compression comp_both_s$s 1 1 $s; done ) &
( for s in 1 2 3; do me 3 cuda $C29 2p9b compression comp_gptq_s$s 0 1 $s; done ) &
wait
fi

if [ ! -e $R/skip_0310_2 ]; then
echo "фаза 2 $(date)" >> $LOG
F9=$R/files_3009; F1=$R/files_0110
( ev 0 cuda $F1/0p1b_compression.rwkvq $C01 0p1b_comp_both_s0
  for a in red comp wprop; do ev 0 cuda $F9/0p1b_$a.rwkvq $C01 0p1b_rtn_$a; ev 0 cuda $F9/0p4b_$a.rwkvq $C04 0p4b_rtn_$a; done
  ev 0 cuda $F1/0p1b_reduction.rwkvq $C01 0p1b_red_gptq; ev 0 cuda $F1/0p4b_reduction.rwkvq $C04 0p4b_red_gptq ) &
( for a in red comp wprop; do ev 1 cuda $F9/1p5b_$a.rwkvq $C15 1p5b_rtn_$a; done
  ev 1 cuda $F1/1p5b_reduction.rwkvq $C15 1p5b_red_gptq ) &
( for a in red comp wprop; do ev 2 cuda $F9/2p9b_$a.rwkvq $C29 2p9b_rtn_$a; done
  ev 2 cuda $F1/2p9b_reduction.rwkvq $C29 2p9b_red_gptq ) &
wait
fi

if [ ! -e $R/skip_0310_3 ]; then
echo "фаза 3 $(date)" >> $LOG
( me 0,1 cuda:0,cuda:1 $C72 7p2b compression comp_both_s0 1 1 ""
  me 0,1 cuda:0,cuda:1 $C72 7p2b reduction red_gptq "" 1 "" ) &
( me 2,3 cuda:0,cuda:1 $C72 7p2b compression comp_gptq_s0 0 1 ""
  for a in red comp wprop; do ev 2,3 cuda:0,cuda:1 $R/files_3009/7p2b_$a.rwkvq $C72 7p2b_rtn_$a; done ) &
wait
fi

if [ ! -e $R/skip_0310_4 ]; then
echo "фаза 4 $(date)" >> $LOG
D4=cuda:0,cuda:1,cuda:2,cuda:3
for a in red comp wprop; do ev 0,1,2,3 $D4 $R/files_3009/13b_$a.rwkvq $C13 13b_rtn_$a; done
me 0,1,2,3 $D4 $C13 13b compression comp_both_s0 1 1 ""
me 0,1,2,3 $D4 $C13 13b compression comp_gptq_s0 0 1 ""
me 0,1,2,3 $D4 $C13 13b reduction red_gptq "" 1 ""
fi
echo "конец $(date)" >> $LOG
