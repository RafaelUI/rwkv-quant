#!/bin/sh
# 04.10: после конца cc7 (chain_0310) -- чужие форматы прибором статьи: GGUF Q4_K_M / Q6_K (1.5B), схема MLX int6 / int4 (1.5B, 2.9B).
# Файлы залиты с Mac в ~/rwkvq/gguf_0310 (md5 сверен заливкой). НЕ убивать sh цепочки. Пропуск: touch ~/rwkvq/skip_0410_ext.
cd ~/rwkvq/rwkv-quant-git
export PYTHONPATH=$HOME/rwkvq/rwkv-quant-git:$HOME/rwkvq
PY="$HOME/venv/bin/python -u"; K=$HOME/rwkvq/ckpt; R=$HOME/rwkvq; LOG=$R/chain_ext_0410.log; G=$R/gguf_0310; FE=$R/fe0310
C15=$K/rwkv7-g1j-1.5b-20260831-ctx16384.pth; C29=$K/rwkv7-g1j-2.9b-20260831-ctx16384.pth
echo "ожидание cc7 $(date)" >> $LOG
while ! grep -q '^конец' $R/chain_0310.log; do sleep 120; done
[ -e $R/skip_0410_ext ] && { echo "пропуск $(date)" >> $LOG; exit 0; }
echo "старт $(date) $(git log --oneline -1 | cut -c1-7)" >> $LOG
gg() { [ -e $G/$2 ] || { echo "нет $2" >> $LOG; return; }; CUDA_VISIBLE_DEVICES=$1 RWKVQ_DEVICE=cuda $PY tests/_sess/gguf_eval_0410.py $G/$2 $3 $FE/ext_$4.json > $FE/ext_$4.log 2>&1; echo "ev ext_$4 код $? $(date)" >> $LOG; }
mm() { [ -e $G/$2 ] || { echo "нет $2" >> $LOG; return; }; CUDA_VISIBLE_DEVICES=$1 RWKVQ_DEVICE=cuda $PY tests/_sess/mlx_eval_0410.py $G/$2 $3 $FE/ext_$4.json > $FE/ext_$4.log 2>&1; echo "ev ext_$4 код $? $(date)" >> $LOG; }
( gg 0 1p5b_Q4_K_M.gguf $C15 1p5b_gguf_Q4_K_M; gg 0 1p5b_Q6_K.gguf $C15 1p5b_gguf_Q6_K; gg 0 2p9b_Q4_K_M.gguf $C29 2p9b_gguf_Q4_K_M; gg 0 2p9b_Q6_K.gguf $C29 2p9b_gguf_Q6_K ) &
( mm 1 1p5b_mlx_int6.npz $C15 1p5b_mlx_int6; mm 1 1p5b_mlx_int4.npz $C15 1p5b_mlx_int4 ) &
( mm 2 2p9b_mlx_int6.npz $C29 2p9b_mlx_int6; mm 2 2p9b_mlx_int4.npz $C29 2p9b_mlx_int4 ) &
wait
echo "конец $(date)" >> $LOG
