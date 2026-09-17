#!/bin/sh
# ВОССТАНОВЛЕНИЕ /tmp ПОСЛЕ РЕБУТА.
#
# Половина гейтов и замеров жёстко зашита на пути в /tmp, а /tmp ребут не
# переживает. Хуже того, act_stats оттуда однажды исчезли БЕЗ ребута, и
# любая сборка пресета после этого молча вырождает AW в обычный поиск
# (закон 15: предупреждение будет, но замер качества пройдёт -- не тот).
# Долговечные копии лежат в ~/Develop/WKV-kvant/artifacts, здесь -- ссылки.
set -e
A="$HOME/Develop/WKV-kvant/artifacts"
ln -sf "$A/act_stats_1p5b.pt"                   /tmp/act_stats_1p5b.pt
ln -sf "$A/act_stats_1p5b_ml.pt"                /tmp/act_stats_1p5b_ml.pt
ln -sf "$A/act_stats_2p9b_ml.pt"                /tmp/act_stats_2p9b_ml.pt
# Прежние скрипты знают ещё одно имя -- _multiling; это тот же файл.
ln -sf "$A/act_stats_1p5b_ml.pt"                /tmp/act_stats_1p5b_multiling.pt
ln -sf "$A/act_stats_2p9b_ml.pt"                /tmp/act_stats_2p9b_multiling.pt
ln -sf "$A/reduction_1p5b_1709.rwkvq"             /tmp/reduction_new.rwkvq
ln -sf "$A/sym_head8_2p9b_1709.rwkvq"           /tmp/reduction_sym_head8_2p9b.rwkvq
# 2.9B нынешним пресетом (26.08): proj/emb/head @8, 2737.3 МБ. Прежний
# файл выше -- пресет 13.08 на шести битах, оставлен как история.
ln -sf "$A/reduction_2p9b_1709.rwkvq"             /tmp/reduction_2p9b_new.rwkvq
ln -sf "$A/compression_1p5b.rwkvq"              /tmp/champion_v2.rwkvq
# 17.09: артефакты 16.08 (reduction_*_preset16aug, reduction_sym_head8_2p9b)
# ПОТЕРЯНЫ; пересобраны tests/_sess/build_artifact.py с суффиксом _1709 (тот
# же рецепт: real_gw, статистика _ml). Имена в /tmp прежние -- гейтам всё
# равно, они сравнивают два пути на одних весах; числам КАЧЕСТВА, снятым на
# файлах 16.08, эти файлы НЕ тождественны (закон 30). Добавлены две ссылки,
# которых скрипт не создавал никогда (долг 4 с 05.09), и эталоны кернелей.
ln -sf "$A/sym_head8_1p5b_1709.rwkvq"     /tmp/reduction_sym_head8.rwkvq
ln -sf "$A/legacy_1p5b_1709.rwkvq"        /tmp/reduction_v2.rwkvq
ln -sf "$A/sym_kernel_ref.npz"            /tmp/sym_kernel_ref.npz
ln -sf "$A/wkv_infer_ref.npz"             /tmp/wkv_infer_ref.npz
echo "восстановлено:"; ls -lL /tmp/act_stats_*.pt /tmp/*.rwkvq /tmp/*_ref.npz 2>&1 | sed 's/^/  /'
