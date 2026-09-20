#!/bin/sh
# 20.09: пресет с o_proj слоя 0 в bf16 -- сквозной контроль С ФАЙЛА (writer->reader->кернель).
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/l0_file_2009.log
cd /Users/s/Develop/rwkv-quant
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
: > $L
C29=/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth; R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
F=$KV/compression_2p9b_l0_2009.rwkvq
echo "##### сборка $(date +%T) своп $(sw)" >> $L
/usr/bin/time -l $PY tests/_sess/build_preset_2009.py $C29 compression $F 2>&1 | grep -E "готово|Error|Traceback|peak memory" >> $L
echo "##### KL с файла $(date +%T) своп $(sw) размер $(stat -f %z $F)" >> $L
RWKVQ_KL_NSEQ=38 RWKVQ_KL_REF=$R29 RWKVQ_KLREAL_OUT=$KV/kl_real_l0_2009.json /usr/bin/time -l $PY tests/_sess/kl_real_path.py $F c2p9b_l0 2>&1 | grep -E "KL|ppl|Error|Traceback|peak memory" >> $L
echo "##### КОНЕЦ $(date +%T) своп $(sw)" >> $L
