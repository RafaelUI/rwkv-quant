#!/bin/bash
# 22.09 ночь: совместная проверка выбора на ПОЛНОЙ лестнице. База compression_nol0 +
# live-ключи (emb 6, o_proj L0 bf16) + ключи правила. Один конфиг -- один процесс.
# Аргументы: <m> <devs> <имя:json-файл-ovr> ...
cd ~/rwkvq; W=~/rwkvq/rwkv-quant/tests/_sess/weakspot_2109.py; PY=~/venv/bin/python
export OMP_NUM_THREADS=4
m=$1; devs=$2; shift 2
source env_$m.sh; export RWKVQ_DEVICE=${devs%%,*} RWKVQ_DEVICES=$devs
for spec in "$@"; do name=${spec%%:*}; f=${spec#*:}
  ovr=$($PY -c "import json,sys; o=json.load(open('$f')); live={'emb.weight':6,'blocks.0.att.output.weight':16}; live.update(o); print(json.dumps(live))")
  echo "$(date +%T) $m $name $f" >> ~/rwkvq/joint_full_2209.log
  $PY $W joint ws_${m}_joint2.json $name "$ovr" >> ws_${m}_joint2.log 2>&1
  tail -2 ws_${m}_joint2.log | head -1 >> ~/rwkvq/joint_full_2209.log
done
