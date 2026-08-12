#!/bin/bash
# Подтверждение Q6_K-раскладки на 2.9B (закон 10). Один конфиг на процесс
# (закон 13), batch=1 -- иначе MPS роняет command buffer.
#
# Шаг 1, статистика активаций для 2.9B (~10 мин, один раз):
#   ~/Develop/tests/venv/bin/python tests/collect_act_stats.py \
#     ~/Develop/WKV-kvant/act_calib_multiling.pt /tmp/act_stats_2p9b_ml.pt ':'
# Шаг 2:
#   bash tests/run_sym_2p9b.sh
set -u
cd "$(dirname "$0")/.."
export RWKVQ_CKPT="${RWKVQ_CKPT:-$HOME/Develop/rwkv7-g1h-2.9b-ctx10240.pth}"
export RWKVQ_ACT_STATS="${RWKVQ_ACT_STATS:-/tmp/act_stats_2p9b_ml.pt}"
export RWKVQ_OUT="${RWKVQ_OUT:-/tmp/sym_cmix_2p9b.json}"
export RWKVQ_BATCH="${RWKVQ_BATCH:-1}"
export RWKVQ_NSEQ="${RWKVQ_NSEQ:-16}"
exec bash tests/run_sym_cmix.sh "${@:-bf16 asym_sb6_aw sym_aw}"
