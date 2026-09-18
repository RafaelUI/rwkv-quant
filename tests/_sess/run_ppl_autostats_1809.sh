#!/bin/sh
# 18.09: перемер COMPRESSION на всех четырёх масштабах с КАЛИБРОВКОЙ ПО
# УМОЛЧАНИЮ (act_stats="auto": корпус репозитория, мультиязычный + код) --
# тем самым рецептом, который получает пользователь. Это снимает расхождение
# +3.224% (act_calib_multiling) против +2.850% (репозиторный корпус) на 1.5B.
# Один масштаб на процесс; 2.9B отдельным прибором.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
AS=$KV/_autostats.py
cd /Users/s/Develop/rwkv-quant
run() {
  name=$1; ck=$2
  echo "===== $name $(date +%T)"
  P=$($PY $AS $ck 2>/dev/null | tail -1)
  echo "статистика: $P ($(ls -l $P 2>/dev/null | awk "{print \$5}") Б)"
  [ -f "$P" ] || { echo "НЕТ СТАТИСТИКИ для $name"; return 1; }
  if [ "$name" = "2p9b" ]; then
    RWKVQ_ACT_STATS=$P RWKVQ_EVAL_JSON=$KV/eval_auto_2p9b_1809.json $PY tests/eval_2p9b_one.py compression 2>&1 | grep -E "ppl за|размер|MB|Error|Traceback"
  else
    RWKVQ_CKPT=$ck RWKVQ_EVAL_JSON=$KV/eval_auto_${name}_1809.json RWKVQ_CONFIGS="bf16 baseline,COMPRESSION" $PY tests/eval_multiling.py $P 2>&1 | grep -E "^конфиг|^bf16 baseline|^COMPRESSION|Error|Traceback"
  fi
  sysctl -n vm.swapusage
}
run 0p1b $KV/rwkv7-g1d-0.1b.pth
run 0p4b /Users/s/rwkv7-g1d-0.4b-ctx8192.pth
run 1p5b $KV/rwkv7-g1h-1.5b-ctx10240.pth
run 2p9b /Users/s/Develop/rwkv7-g1h-2.9b-ctx10240.pth
echo "[$(date +%T)] КОНЕЦ_AUTO"
