#!/bin/sh
# 21.09: варианты правила на Mac реальным путём. ppl/KL с файла рядом с нынешним
# пресетом (compression_<s>_e6_2109 = o_proj L0 bf16 + emb 6), затем декод ABBA.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant; S=$KV/sweep_2109; L=$KV/rule_mac_2109.log
cd /Users/s/Develop/rwkv-quant
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
: > $L
for s in 0p1b 0p4b; do
  [ $s = 0p1b ] && C=$KV/rwkv7-g1d-0.1b.pth R=$KV/kl_ref_rwkv7-g1d-0p1b_eval_corpus_multiling_38x512_fp32.npy
  [ $s = 0p4b ] && C=/Users/s/rwkv7-g1d-0.4b-ctx8192.pth R=$KV/kl_ref_rwkv7-g1d-0p4b-ctx8192_eval_corpus_multiling_38x512_fp32.npy
  for t in 10 5; do
    echo "##### сборка $s tau$t $(date +%T)" >> $L
    $PY tests/_sess/build_rule_2109.py $C $S/ws_${s}_ovr_tau$t.json $KV/rule_${s}_tau$t.rwkvq 2>&1 | grep -E "готово|Error|Trace" >> $L
  done
  for arm in live:$KV/compression_${s}_e6_2109.rwkvq tau10:$KV/rule_${s}_tau10.rwkvq tau5:$KV/rule_${s}_tau5.rwkvq; do
    n=${arm%%:*}; f=${arm#*:}
    echo "##### $s $n $(date +%T) размер $(stat -f %z $f)" >> $L
    RWKVQ_KL_REF=$R $PY tests/_sess/ce_real_path_1909.py $f $n $KV/ce_rule_${s}_2109.json 2>&1 | grep -E "KL|ppl|Error|Trace" | tail -n 1 >> $L
  done
  echo "##### декод ABBA $s live(A) vs tau5(B) $(date +%T) своп $(sw)" >> $L
  $PY tests/_sess/bench_files_ab_2009.py $KV/compression_${s}_e6_2109.rwkvq $KV/rule_${s}_tau5.rwkvq 16 32 2>&1 | tail -n 4 >> $L
  echo "##### своп после $(sw)" >> $L
done
echo "##### КОНЕЦ $(date +%T)" >> $L
