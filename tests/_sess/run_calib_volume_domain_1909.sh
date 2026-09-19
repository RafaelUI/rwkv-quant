#!/bin/sh
# 19.09: объём против домена в калибровке AW. Три плеча на масштаб, KL на 38 окнах:
#   repo       -- штатный файл (act_stats=auto, 23x511 = 11753 токена, широкий домен)
#   reposmall  -- тот же корпус, окна укорочены до 379: 8694 токена, тот же состав
#   narrow     -- act_calib_multiling (17x511 = 8687, домен замера, не пересекается)
# repo vs reposmall = эффект объёма; reposmall vs narrow = эффект домена при равном объёме.
# Контроль детерминизма: пересборка repo-плеча 0.4B обязана совпасть с файлом побайтово.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant; A=$KV/artifacts
L=$KV/calib_volume_domain_1909.log
cd /Users/s/Develop/rwkv-quant
: > $L
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
step() { echo "##### $* $(date +%T) своп $(sw)" >> $L; }
C04=/Users/s/rwkv7-g1d-0.4b-ctx8192.pth; C15=$KV/rwkv7-g1h-1.5b-ctx10240.pth
R04=$KV/kl_ref_rwkv7-g1d-0p4b-ctx8192_eval_corpus_multiling_38x512_fp32.npy
R15=$KV/kl_ref_rwkv7-g1h-1p5b-ctx10240_eval_corpus_multiling_38x512_fp32.npy
S04=$HOME/.cache/rwkv-quant/act_stats/act_cb9522cbd3ea0e4c.pt
kl() { step KL $1 $3; RWKVQ_KL_NSEQ=38 RWKVQ_KL_REF=$2 RWKVQ_KLREAL_OUT=$KV/calibvd_$1_1909.json $PY tests/_sess/kl_real_path.py $4 $3 >> $L 2>&1; }
# статистика reposmall
step stats reposmall 0p4b; $PY tests/_sess/collect_reposmall_1909.py $C04 $A/act_stats_0p4b_reposmall_1909.pt >> $L 2>&1
step stats reposmall 1p5b; $PY tests/_sess/collect_reposmall_1909.py $C15 $A/act_stats_1p5b_reposmall_1909.pt >> $L 2>&1
# контроль детерминизма сборки
step build repo_rebuild 0p4b; $PY tests/_sess/build_calib_arm_1909.py $C04 $S04 $A/compression_0p4b_reporebuild_1909.rwkvq >> $L 2>&1
$PY tests/_sess/cmp_rwkvq_1909.py $KV/compression_0p4b.rwkvq $A/compression_0p4b_reporebuild_1909.rwkvq >> $L 2>&1
# сборки
step build 0p4b; $PY tests/_sess/build_calib_arm_1909.py $C04 $A/act_stats_0p4b_reposmall_1909.pt $A/compression_0p4b_reposmall_1909.rwkvq >> $L 2>&1
$PY tests/_sess/build_calib_arm_1909.py $C04 $A/act_stats_0p4b_calibml.pt $A/compression_0p4b_narrow_1909.rwkvq >> $L 2>&1
step build 1p5b; $PY tests/_sess/build_calib_arm_1909.py $C15 $A/act_stats_1p5b_reposmall_1909.pt $A/compression_1p5b_reposmall_1909.rwkvq >> $L 2>&1
$PY tests/_sess/build_calib_arm_1909.py $C15 $KV/act_stats_calibml.pt $A/compression_1p5b_narrow_1909.rwkvq >> $L 2>&1
# KL, три плеча на масштаб в один json (пары бутстрэпятся парно в --report)
kl 0p4b $R04 repo $KV/compression_0p4b.rwkvq
kl 0p4b $R04 reposmall $A/compression_0p4b_reposmall_1909.rwkvq
kl 0p4b $R04 narrow $A/compression_0p4b_narrow_1909.rwkvq
kl 1p5b $R15 repo $KV/compression_v2_cand.rwkvq
kl 1p5b $R15 reposmall $A/compression_1p5b_reposmall_1909.rwkvq
kl 1p5b $R15 narrow $A/compression_1p5b_narrow_1909.rwkvq
for s in 0p4b 1p5b; do step report $s; RWKVQ_KLREAL_OUT=$KV/calibvd_${s}_1909.json $PY tests/_sess/kl_real_path.py --report >> $L 2>&1; done
step КОНЕЦ
