#!/bin/bash
# 22.09 ночь: файлы симметричного правила (g1j) по одному + KL самого файла на 38 окнах.
cd ~/rwkvq/rwkv-quant; PY=~/venv/bin/python; O=~/rwkvq/sym_files; mkdir -p $O; L=$O/run.log
export RWKVQ_CORPUS=~/rwkvq/eval_corpus_multiling.pt OMP_NUM_THREADS=8
C1p5b=~/rwkvq/ckpt/rwkv7-g1j-1.5b-20260831-ctx16384.pth; C2p9b=~/rwkvq/ckpt/rwkv7-g1j-2.9b-20260831-ctx16384.pth
R1p5b=~/rwkvq/kl_ref_1p5b_g1j_38x512_fp32.npy; R2p9b=~/rwkvq/kl_ref_2p9b_g1j_38x512_fp32.npy
g=0
for s in 1p5b 2p9b; do for t in 6 7 5; do
  C=C$s; R=R$s; f=$O/${s}_sym$t.rwkvq
  echo "$(date +%T) сборка $s sym$t" >> $L
  $PY tests/_sess/build_rule_a_2209.py ${!C} ~/rwkvq/sym_${s}_tau$t.json $f > $O/build_${s}_sym$t.log 2>&1
  grep готово $O/build_${s}_sym$t.log >> $L
  RWKVQ_CKPT=${!C} RWKVQ_KL_REF=${!R} $PY tests/_sess/kl_file_2209.py $f cuda:$g $O/kl_${s}_sym$t.json > $O/kl_${s}_sym$t.log 2>&1 &
  g=$(( (g+1) % 4 ))
done; done
wait
cd $O && md5sum *.rwkvq > MD5SUMS
echo "$(date +%T) КОНЕЦ" >> $L
