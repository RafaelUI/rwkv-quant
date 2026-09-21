#!/bin/sh
# 21.09: строки README COMPRESSION после emb 5->6 (решение владельца: всегда).
# Сборка публичным путём, ppl и KL С ФАЙЛА (ce_real_path_1909), прежний файл
# (o_proj слоя 0, emb 5) тем же прибором рядом. Файлов 2.9B на диске нет --
# контроль там: число в памяти 21.09 (e6 0.024885).
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/readme_e6_2109.log
cd /Users/s/Develop/rwkv-quant
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
: > $L
one() {  # масштаб чекпоинт эталон [старый]
  s=$1; F=$KV/compression_${s}_e6_2109.rwkvq; R=$3
  echo "##### сборка $s $(date +%T) своп $(sw)" >> $L
  $PY tests/_sess/build_preset_2009.py $2 compression $F 2>&1 | grep -E "готово|Error|Traceback" >> $L
  for arm in "compression_e6 $F" ${4:+"compression_l0 $4"}; do
    set -- $arm
    echo "##### $s $1 $(date +%T) своп $(sw) размер $(stat -f %z $2)" >> $L
    RWKVQ_KL_REF=$R $PY tests/_sess/ce_real_path_1909.py $2 $1 $KV/ce_e6_${s}_2109.json 2>&1 | grep -E "KL|ppl|Error|Traceback" | tail -2 >> $L
  done
}
one 0p1b $KV/rwkv7-g1d-0.1b.pth $KV/kl_ref_rwkv7-g1d-0p1b_eval_corpus_multiling_38x512_fp32.npy $KV/compression_0p1b_l0_2009.rwkvq
one 0p4b /Users/s/rwkv7-g1d-0.4b-ctx8192.pth $KV/kl_ref_rwkv7-g1d-0p4b-ctx8192_eval_corpus_multiling_38x512_fp32.npy $KV/compression_0p4b_l0_2009.rwkvq
one 1p5b $KV/rwkv7-g1h-1.5b-ctx10240.pth $KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy $KV/compression_1p5b_l0_2009.rwkvq
one 2p9b /Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth $KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
echo "##### КОНЕЦ $(date +%T) своп $(sw)" >> $L
