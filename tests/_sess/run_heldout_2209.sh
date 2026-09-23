#!/bin/bash
# 22.09 ночь: KL на 30 ОТЛОЖЕННЫХ окнах fake-путём на НЕСКОЛЬКИХ картах (7.2B/13.3B в fp32
# не входят в одну карту -> kl_file_2209 не годится). Группы по G окон, один конфиг --
# один процесс. База compression_nol0 + live (emb 6, oL0 bf16) + ovr.
#   run_heldout_2209.sh <m> <devs> <G> <имя:json> ...   -> ws_<m>_ho.json (joint.<имя>_g<k>)
cd ~/rwkvq; W=~/rwkvq/rwkv-quant/tests/_sess/weakspot_2109.py; PY=~/venv/bin/python
export OMP_NUM_THREADS=4
m=$1; devs=$2; G=$3; shift 3
source env_$m.sh; export RWKVQ_DEVICE=${devs%%,*} RWKVQ_DEVICES=$devs
HO=$($PY -c "print(','.join(str(i) for i in range(38) if i not in (0,6,13,20,24,28,30,35)))")
IFS=, read -ra A <<< "$HO"
for spec in "$@"; do name=${spec%%:*}; f=${spec#*:}
  ovr=$($PY -c "import json; o=json.load(open('$f')); live={'emb.weight':6,'blocks.0.att.output.weight':16}; live.update(o); print(json.dumps(live))")
  for ((k=0; k*G<${#A[@]}; k++)); do
    win=$(IFS=,; echo "${A[*]:k*G:G}")
    RWKVQ_SWEEP_WIN=$win $PY $W joint ws_${m}_ho.json ${name}_g$k "$ovr" >> ws_${m}_ho.log 2>&1
    echo "$(date +%T) $m $name g$k $(tail -2 ws_${m}_ho.log | head -1)" >> ~/rwkvq/heldout_2209.log
  done
done
