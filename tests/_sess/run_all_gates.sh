#!/bin/sh
# Все гейты проекта, ОДИН гейт на процесс (закон 2), последовательно.
# Красный гейт перезапускается с RWKVQ_GW_DQ_REF=1 (рабочий деквант как до
# 17.09): красный и там -- дефект был до правки, зелёный -- виновата правка.
# Отдельно: гейт кернеля с RWKVQ_GW_DQ_REF=1 ОБЯЗАН покраснеть на маршруте.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
KV=/Users/s/Develop/WKV-kvant
D=$KV/gates_1709
mkdir -p $D
cd /Users/s/Develop/rwkv-quant
run() {
  name=$1; shift
  t0=$(date +%s)
  /usr/bin/time -l "$@" > $D/$name.log 2>&1
  rc=$?
  t1=$(date +%s)
  pk=$(grep "peak memory footprint" $D/$name.log | tr -s " " | cut -d" " -f2)
  echo "$name rc=$rc $((t1-t0))s peak=${pk}B swap=$(sysctl -n vm.swapusage | awk "{print \$6}") $(vm_stat | grep Swapouts | tr -s " ")"
  if [ $rc -ne 0 ]; then
    RWKVQ_GW_DQ_REF=1 /usr/bin/time -l "$@" > $D/$name.REF.log 2>&1
    echo "   -> с RWKVQ_GW_DQ_REF=1: rc=$?"
  fi
}
echo "старт $(date) $(sysctl -n vm.swapusage)"
for f in tests/test_*.py; do
  n=$(basename $f .py)
  [ -s $f ] || { echo "$n ПУСТОЙ ФАЙЛ, пропуск"; continue; }
  case $n in
    test_k3_from_canonical) echo "$n пропуск: нет сайдкара (путь GwQuantLinear.__init__, не _dequant_w)";;
    test_manifest_selfdesc|test_mlx_affine_repack) run $n $PY $f $KV/compression_0p1b.rwkvq;;
    test_rwkvq_container) run $n $PY $f $KV/compression_0p1b.rwkvq $D/c0p1b.st_roundtrip; rm -f $D/c0p1b.st_roundtrip;;
    test_wkv_var_model) run ${n}_ru60m $PY $f ru60m; run ${n}_1p5b $PY $f 1.5b;;
    *) run $n $PY $f;;
  esac
done
rm -f $KV/compression_0p1b.rwkvq.selfdesc
RWKVQ_GW_DQ_REF=1 $PY tests/test_gw_dequant_kernel_parity.py $KV/compression_0p1b.rwkvq > $D/route_mutation.log 2>&1
echo "мутация маршрута (REF=1) rc=$? -- обязан быть НЕнулевым; $(grep -c "маршрут _dequant_w не совпал" $D/route_mutation.log) совпадений ассерта"
echo "КОНЕЦ $(date)"
