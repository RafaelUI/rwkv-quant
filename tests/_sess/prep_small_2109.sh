#!/bin/bash
# act-статистика и fp32-эталоны 38x512 для 0.1B/0.4B g1d на сервере
cd ~/rwkvq/rwkv-quant; export PYTHONPATH=$HOME/rwkvq
for s in 0p1b 0p4b; do
  [ $s = 0p1b ] && C=~/rwkvq/ckpt/rwkv7-g1d-0.1b-20260129-ctx8192.pth || C=~/rwkvq/ckpt/rwkv7-g1d-0.4b-20260210-ctx8192.pth
  SIG=$(RWKVQ_DEVICE=cuda:3 ~/venv/bin/python -c "
import sys; sys.path.insert(0,\".\")
from rwkv_quant.calibration import act_stats as A
s,sig=A.collect(\"$C\", \"$HOME/rwkvq/rwkv_vocab_v20230424.txt\", verbose=False); print(sig)" | tail -n 1)
  echo "$s act $SIG"
  cat > ~/rwkvq/env_$s.sh <<EOT
export PYTHONPATH=\$HOME/rwkvq
export RWKVQ_DEVICE=cuda
export RWKVQ_REF_DEVICE=cuda
export RWKVQ_CKPT=$C
export RWKVQ_CORPUS=\$HOME/rwkvq/eval_corpus_multiling.pt
export RWKVQ_ACT_STATS=\$HOME/.cache/rwkv-quant/act_stats/act_$SIG.pt
export RWKVQ_KL_NSEQ=38
export RWKVQ_KL_REF=\$HOME/rwkvq/kl_ref_${s}_g1d_38x512_fp32.npy
export RWKVQ_KL_OUT=\$HOME/rwkvq/kl_subgroups_${s}_g1d.json
EOT
  source ~/rwkvq/env_$s.sh
  RWKVQ_DEVICE=cuda:3 RWKVQ_REF_DEVICE=cuda:3 ~/venv/bin/python tests/ablate_subgroups.py --ref 2>&1 | tail -n 2
  ls -la $RWKVQ_KL_REF
done
echo PREP_DONE
