#!/bin/bash
# Q6_K-подобная симметричная раскладка против нашей асимметричной, на cmix.
# Открытый вопрос 2 из NEXT_SESSION, закрываемый замером на реальной модели.
#
#   bash tests/run_sym_cmix.sh                 # все режимы
#   bash tests/run_sym_cmix.sh sym_aw          # только этот
#   RWKVQ_CKPT=~/Develop/rwkv7-g1h-2.9b-ctx10240.pth \
#     RWKVQ_BATCH=1 RWKVQ_ACT_STATS=/tmp/act_stats_2p9b_ml.pt \
#     bash tests/run_sym_cmix.sh               # на 2.9B (закон 10)
#
# Сводка в любой момент, не мешая прогону:
#   python tests/ablate_sym_cmix.py --report
set -u
cd "$(dirname "$0")/.."
export RWKVQ_ACT_STATS="${RWKVQ_ACT_STATS:-/tmp/act_stats_1p5b_ml.pt}"
P="${PY:-$HOME/Develop/tests/venv/bin/python}"

if [ ! -f "$RWKVQ_ACT_STATS" ]; then
  echo "нет act_stats: $RWKVQ_ACT_STATS"
  echo "собрать (~5 мин):  $P tests/collect_act_stats.py \\"
  echo "    ~/Develop/WKV-kvant/act_calib_multiling.pt $RWKVQ_ACT_STATS ':'"
  exit 1
fi

MODES=("$@")
[ ${#MODES[@]} -eq 0 ] && MODES=(bf16 asym_sb6_aw sym_aw asym_sb6_search sym)
for m in "${MODES[@]}"; do
  echo "=================== $m   (своп: $(sysctl -n vm.swapusage))"
  /usr/bin/time -l "$P" -u tests/ablate_sym_cmix.py "$m" 2>&1 \
    | grep -v -e RuntimeWarning -e "ex2 = " -e "^\s*$" \
    | grep -v -e voluntary -e reclaims -e faults -e swaps -e messages \
              -e signals -e switches -e "block input" -e "block output" \
              -e "average " -e "maximum resident"
  sleep 5
done
echo "=================== ГОТОВО"
"$P" tests/ablate_sym_cmix.py --report
