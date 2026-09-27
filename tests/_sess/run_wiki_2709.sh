#!/bin/sh
PY=/Users/s/Develop/rwkv-metal/.venv/bin/python
D=/Users/s/Develop/WKV-kvant/g1j_2309
CK=/Users/s/Develop/WKV-kvant/rwkv7-g1j-1.5b-20260831-ctx16384.pth
cd /Users/s/Develop/rwkv-quant
echo "##### гейт ref $(date +%T)" > $D/run_wiki_2709.log
$PY tests/test_ref_storage_dtype.py >> $D/run_wiki_2709.log 2>&1
echo "##### autopick на Википедии $(date +%T)" >> $D/run_wiki_2709.log
RWKVQ_EVAL_TEXT=/Users/s/Develop/WKV-kvant/eval_text_heldout.pt $PY tests/_sess/budget_eval_all_2409.py $CK $D/wiki_autopick_2709.json wprop=$D/sel_wprop_b05.json >> $D/run_wiki_2709.log 2>&1
echo "##### GPTQ + wprop на Википедии $(date +%T)" >> $D/run_wiki_2709.log
RWKVQ_EVAL_TEXT=/Users/s/Develop/WKV-kvant/eval_text_heldout.pt $PY tests/_sess/gptq_proto_2709.py $CK $D/gptq_wprop_1p5b_wiki_2709.json 48 0.01 $D/sel_wprop_b05.json >> $D/run_wiki_2709.log 2>&1
echo "##### DONE $(date +%T)" >> $D/run_wiki_2709.log
