#!/bin/sh
# 07.10: гейты с аргументами на файлах (п. 1 сессии). Старые файлы files_0110 (читалка на прежних
# манифестах) + свежие сборки 0.1B нынешним кодом (писатель после правок 06.10: runtime.lora_q,
# подпись кеша). Каждый гейт -- свой журнал; сводка кодов в $O/summary.txt.
P=/Users/s/Develop/rwkv-metal/.venv/bin/python
R=$HOME/Develop/rwkv-quant
F=$HOME/Develop/WKV-kvant/files_0110
CK=$HOME/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth
TOK=$HOME/Develop/WKV-kvant/rwkv_vocab.txt
O=/tmp/g1_0710
mkdir -p $O; cd $R; : > $O/summary.txt
echo "HEAD $(git rev-parse --short HEAD); swap до: $(sysctl -n vm.swapusage)" >> $O/summary.txt
run() { # имя журнала, команда...
  n=$1; shift
  "$@" > $O/$n.log 2>&1; c=$?
  echo "код $c  $n  :: $*" >> $O/summary.txt
}
for pr in compression reduction; do
  run build_$pr $P -c "
import rwkv_quant, sys
from rwkv_quant.api import quantize
print(rwkv_quant.__file__)
quantize('$CK', '$O/0p1b_${pr}_new.rwkvq', preset='$pr', tokenizer='$TOK', gptq=False, autopick=False)
"
done
for f in $F/0p1b_compression $F/0p1b_reduction $O/0p1b_compression_new $O/0p1b_reduction_new $F/0p4b_compression $F/0p4b_reduction; do
  b=$(basename $f)
  run selfdesc_$b  $P tests/test_manifest_selfdesc.py $f.rwkvq
  run container_$b $P tests/test_rwkvq_container.py $f.rwkvq $O/rt_$b.rwkvq
  run affine_$b    $P tests/test_mlx_affine_repack.py $f.rwkvq
done
for f in $F/1p5b_compression $F/1p5b_reduction; do
  b=$(basename $f)
  run selfdesc_$b  $P tests/test_manifest_selfdesc.py $f.rwkvq
  run affine_$b    $P tests/test_mlx_affine_repack.py $f.rwkvq
done
for f in $F/0p1b_compression $O/0p1b_compression_new $F/0p1b_reduction $F/0p4b_compression; do
  b=$(basename $f)
  run export_$b $P -m rwkv_quant.formats.export_mlx $f.rwkvq $O/sc_$b
  run k3_$b     $P tests/test_k3_from_canonical.py $f.rwkvq $O/sc_$b
done
run wkv_var_1p5b $P tests/test_wkv_var_model.py 1.5b
run wkv_var_ru60m $P tests/test_wkv_var_model.py ru60m
echo "swap после: $(sysctl -n vm.swapusage)" >> $O/summary.txt
echo DONE >> $O/summary.txt
