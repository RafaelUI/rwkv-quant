#!/bin/sh
# чистая пара на 0.1B: те же пресеты текущим кодом БЕЗ o_proj слоя 0 в bf16
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/readme_l0_2009.log
cd /Users/s/Develop/rwkv-quant
R=$KV/kl_ref_rwkv7-g1d-0p1b_eval_corpus_multiling_38x512_fp32.npy
for p in reduction compression; do
  F=$KV/${p}_0p1b_nol0_2009.rwkvq
  $PY tests/_sess/build_preset_2009.py $KV/rwkv7-g1d-0.1b.pth ${p}_nol0 $F 2>&1 | grep -E "Error|Traceback" >> $L
  echo "##### 0p1b ${p}_nol0 $(date +%T) размер $(stat -f %z $F)" >> $L
  RWKVQ_KL_REF=$R $PY tests/_sess/ce_real_path_1909.py $F ${p}_nol0 $KV/ce_l0_0p1b_2009.json 2>&1 | grep -E "KL|ppl|Error|Traceback" | tail -2 >> $L
done
echo "##### КОНЕЦ-2 $(date +%T)" >> $L
