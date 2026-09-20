#!/bin/sh
# 20.09: строки README после o_proj слоя 0 в bf16 -- 4 масштаба x 2 пресета,
# сборка публичным путём, ppl и KL С ФАЙЛА (ce_real_path_1909), старые файлы
# тем же прибором рядом (парность по окнам).
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/readme_l0_2009.log
cd /Users/s/Develop/rwkv-quant
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
: > $L
one() {  # масштаб чекпоинт эталон старый_red старый_comp
  s=$1
  for p in reduction compression; do
    F=$KV/${p}_${s}_l0_2009.rwkvq
    echo "##### сборка $s $p $(date +%T) своп $(sw)" >> $L
    $PY tests/_sess/build_preset_2009.py $2 $p $F 2>&1 | grep -E "готово|Error|Traceback" >> $L
    echo "   размер $(stat -f %z $F)" >> $L
  done
  for arm in "reduction_old $4" "reduction_l0 $KV/reduction_${s}_l0_2009.rwkvq" "compression_old $5" "compression_l0 $KV/compression_${s}_l0_2009.rwkvq"; do
    set -- $arm
    echo "##### $s $1 $(date +%T) своп $(sw) размер $(stat -f %z $2)" >> $L
    RWKVQ_KL_REF=$R $PY tests/_sess/ce_real_path_1909.py $2 $1 $KV/ce_l0_${s}_2009.json 2>&1 | grep -E "KL|ppl|Error|Traceback" | tail -2 >> $L
  done
}
R=$KV/kl_ref_rwkv7-g1d-0p1b_eval_corpus_multiling_38x512_fp32.npy
one 0p1b $KV/rwkv7-g1d-0.1b.pth $R $KV/artifacts/reduction_0p1b_1709.rwkvq $KV/compression_0p1b.rwkvq
R=$KV/kl_ref_rwkv7-g1d-0p4b-ctx8192_eval_corpus_multiling_38x512_fp32.npy
one 0p4b /Users/s/rwkv7-g1d-0.4b-ctx8192.pth $R $KV/artifacts/reduction_0p4b_1709.rwkvq $KV/compression_0p4b.rwkvq
R=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
one 1p5b $KV/rwkv7-g1h-1.5b-ctx10240.pth $R $KV/artifacts/reduction_1p5b_1709.rwkvq $KV/compression_v2_cand.rwkvq
R=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
one 2p9b /Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth $R $KV/artifacts/reduction_2p9b_1709.rwkvq $KV/compression_2p9b.rwkvq
echo "##### КОНЕЦ $(date +%T) своп $(sw)" >> $L
