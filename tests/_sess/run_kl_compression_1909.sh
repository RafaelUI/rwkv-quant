#!/bin/sh
# 19.09: KL(fp32 ‖ COMPRESSION) на реальном пути, 8x512, все четыре масштаба --
# та же схема, что у строк REDUCTION в README (kl_real_path.py). Эталоны 0.1B и
# 0.4B пересобираются на 8 окон (1.07 ГБ вместо 5.1); 1.5B и 2.9B берутся 38-оконные
# (первые 8 окон те же). Контроль эталона: MLX-путь без квантования (bf16-веса)
# обязан дать KL порядка 1e-6 (закон 38). Один прогон на процесс.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
L=$KV/kl_compression_1909.log
cd /Users/s/Develop/rwkv-quant
: > $L
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
ref() {  # масштаб чекпоинт
  R=$KV/kl_ref_$1_eval_corpus_multiling_8x512_fp32.npy
  echo "##### эталон $1 $(date +%T) своп $(sw)" >> $L
  RWKVQ_CKPT=$2 RWKVQ_KL_NSEQ=8 RWKVQ_KL_REF=$R /usr/bin/time -l $PY tests/ablate_subgroups.py --ref >> $L 2>&1
  echo "  своп после $(sw); файл $(ls -l $R 2>/dev/null | awk '{print $5}') Б" >> $L
}
kl() {  # масштаб эталон файл_или_bf16 имя [чекпоинт]
  echo "##### KL $1 $4 $(date +%T) своп $(sw)" >> $L
  RWKVQ_CKPT=$5 RWKVQ_KL_NSEQ=8 RWKVQ_KL_REF=$2 RWKVQ_KLREAL_OUT=$KV/kl_compression_$1_1909.json \
    /usr/bin/time -l $PY tests/_sess/kl_real_path.py $3 $4 >> $L 2>&1
  echo "  своп после $(sw)" >> $L
}
C01=$KV/rwkv7-g1d-0.1b.pth; C04=/Users/s/rwkv7-g1d-0.4b-ctx8192.pth
C15=$KV/rwkv7-g1h-1.5b-ctx10240.pth; C29=/Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth
R01=$KV/kl_ref_0p1b_eval_corpus_multiling_8x512_fp32.npy
R04=$KV/kl_ref_0p4b_eval_corpus_multiling_8x512_fp32.npy
R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
R29=$KV/kl_ref_rwkv7-g1h-2p9b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
ref 0p1b $C01
ref 0p4b $C04
kl 0p1b $R01 bf16 control_bf16 $C01
kl 0p1b $R01 $KV/compression_0p1b.rwkvq compression
kl 0p4b $R04 bf16 control_bf16 $C04
kl 0p4b $R04 $KV/compression_0p4b.rwkvq compression
kl 1p5b $R15 bf16 control_bf16 $C15
kl 1p5b $R15 $KV/compression_v2_cand.rwkvq compression
kl 2p9b $R29 $KV/compression_2p9b.rwkvq compression
echo "##### КОНЕЦ $(date +%T)" >> $L
