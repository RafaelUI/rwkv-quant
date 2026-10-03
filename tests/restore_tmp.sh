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
# 03.10: артефакты _1709 (*.rwkvq) УДАЛЕНЫ уборкой 29.09. Замена:
#  - reduction_new / reduction_2p9b_new / champion_v2 -- ЖИВЫЕ файлы умолчаний quantize()
#    01.10 (files_0110: g1j 1.5B/2.9B reduction, 1.5B compression; с GPTQ и autopick).
#    Гейтам это всё равно -- они сравнивают два пути на одних весах; числам качества,
#    снятым на файлах _1709, эти файлы НЕ тождественны (закон 30).
#  - sym_head8 / legacy -- пересобраны tests/_sess/build_artifact.py нынешним кодом с
#    суффиксом _0310 (RWKVQ_ART_SUFFIX=_0310), тот же рецепт, БЕЗ GPTQ. Размер отличается от
#    _1709 на 24 / 16 байт (манифест).
F="$HOME/Develop/WKV-kvant/files_0110"
ln -sf "$F/1p5b_reduction.rwkvq"          /tmp/reduction_new.rwkvq
ln -sf "$F/2p9b_reduction.rwkvq"          /tmp/reduction_2p9b_new.rwkvq
ln -sf "$F/1p5b_compression.rwkvq"        /tmp/champion_v2.rwkvq
ln -sf "$A/sym_head8_1p5b_0310.rwkvq"     /tmp/reduction_sym_head8.rwkvq
ln -sf "$A/legacy_1p5b_0310.rwkvq"        /tmp/reduction_v2.rwkvq
rm -f /tmp/reduction_sym_head8_2p9b.rwkvq   # 2.9B sym_head8 не пересобран: ни один гейт его не читает
ln -sf "$A/sym_kernel_ref.npz"            /tmp/sym_kernel_ref.npz
ln -sf "$A/wkv_infer_ref.npz"             /tmp/wkv_infer_ref.npz
echo "восстановлено:"; ls -lL /tmp/act_stats_*.pt /tmp/*.rwkvq /tmp/*_ref.npz 2>&1 | sed 's/^/  /'
