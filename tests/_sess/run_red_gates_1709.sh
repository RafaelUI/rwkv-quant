#!/bin/sh
# 17.09: заморозка эталонов кернелей на чистых деревьях (rwkv-quant 94c8bf3,
# rwkv-metal 0bdfbe7 -- sym-кернель и WKV-infer с прошлого зелёного не менялись)
# и перепрогон гейтов, красных из-за потерянных артефактов. Один на процесс.
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
D=/Users/s/Develop/WKV-kvant/gates_1709
cd /Users/s/Develop/rwkv-quant
run() {
  name=$1; shift
  t0=$(date +%s)
  "$@" > $D/$name.1709b.log 2>&1
  echo "$name rc=$? $(($(date +%s)-t0))s swap=$(sysctl -n vm.swapusage | cut -d= -f3 | cut -d" " -f2)"
}
echo "старт $(date)"
run freeze_sym_fuse $PY tests/test_sym_fuse_parity.py --freeze
run freeze_wkv_infer $PY tests/test_wkv_infer_parity.py --freeze
for n in test_sym_fuse_parity test_wkv_infer_parity test_dense_load_parity test_emb_gather_parity test_sym_dequant_fp32 test_sym_dequant_kernel test_lora_quant_parity test_gw_nb_parity test_gw_kernel_int6 test_dequant_band_parity; do
  run $n $PY tests/$n.py
done
echo "КОНЕЦ $(date)"
