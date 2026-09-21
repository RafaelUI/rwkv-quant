#!/bin/bash
# $1 = масштаб (env_$1.sh), $2 = конфиг; 4 шарда на 4 карты
source ~/rwkvq/env_$1.sh
for i in 0 1 2 3; do
  RWKVQ_DEVICE=cuda:$i nohup ~/venv/bin/python ~/rwkvq/rwkv-quant/tests/_sess/only_sweep_2109.py $2 $i/4 ~/rwkvq/sweep_$1_$2_s$i.json > ~/rwkvq/sweep_$1_$2_s$i.log 2>&1 &
done
wait
echo ВСЕ_ШАРДЫ > ~/rwkvq/sweep_$1_$2.done
