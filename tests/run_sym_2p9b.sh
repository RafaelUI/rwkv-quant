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
# Закон 15, вшитый в скрипт, а не оставленный комментарием: статистика
# обязана принадлежать ТОМУ чекпоинту, который меряем. Иначе _get_ex2
# вернёт None, sym_aw тихо выродится в sym, и прогон даст не тот ответ,
# что заказан. Файл с "2p9b" в имени и содержимым 1.5B уже случался.
"${PY:-$HOME/Develop/tests/venv/bin/python}" - "$RWKVQ_CKPT" "$RWKVQ_ACT_STATS" <<'CHECK' || exit 1
import sys, torch
ckpt, stats = sys.argv[1], sys.argv[2]
import os
if not os.path.exists(stats):
    print(f"НЕТ ФАЙЛА СТАТИСТИКИ: {stats}")
    print("Собрать (RWKVQ_CKPT обязателен, иначе возьмётся 1.5B):")
    print(f"  RWKVQ_CKPT={ckpt} \\")
    print("  ~/Develop/tests/venv/bin/python tests/collect_act_stats.py \\")
    print(f"    ~/Develop/WKV-kvant/act_calib_multiling.pt {stats} ':'")
    sys.exit(1)
sd = torch.load(ckpt, map_location="cpu", mmap=True)
want = int(sd["emb.weight"].shape[1])
d = torch.load(stats)
k = [x for x in d if x.endswith("ffn.key.weight")][0]
got = int(d[k].numel())
if got != want:
    print(f"СТАТИСТИКА ОТ ДРУГОЙ МОДЕЛИ: {stats} снята для {got} каналов, "
          f"а у {os.path.basename(ckpt)} их {want}.\nПересобрать с "
          f"RWKVQ_CKPT={ckpt}")
    sys.exit(1)
print(f"статистика сходится: {got} каналов, {len(d)} тензоров")
CHECK

if [ $# -eq 0 ]; then
  set -- bf16 asym_sb6_aw sym_aw
fi
exec bash tests/run_sym_cmix.sh "$@"
