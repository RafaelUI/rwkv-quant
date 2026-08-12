#!/bin/bash
# Подтверждение Q6_K-раскладки на 2.9B (закон 10). Один конфиг на процесс
# (закон 13), batch=1 -- иначе MPS роняет command buffer.
#
# Шаг 1, статистика активаций ДЛЯ 2.9B (~10 мин, один раз).
# RWKVQ_CKPT ОБЯЗАТЕЛЕН: без него collect_act_stats берёт 1.5B по
# умолчанию и молча кладёт статистику 1.5B в файл с именем 2.9B. Дальше
# _get_ex2 увидит 2048 каналов вместо 2560, ПРЕДУПРЕДИТ и вернёт None --
# то есть sym_aw тихо выродится в sym, и замер получится не тот, что
# заказан (закон 15).
#   RWKVQ_CKPT=~/Develop/rwkv7-g1h-2.9b-ctx10240.pth \
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
# Кавычки вокруг "${@:-...}" склеивали умолчание в ОДИН аргумент, и
# скрипт честно отвечал "неизвестный режим bf16 asym_sb6_aw sym_aw".
if [ $# -eq 0 ]; then
  set -- bf16 asym_sb6_aw sym_aw
fi
exec bash tests/run_sym_cmix.sh "$@"
