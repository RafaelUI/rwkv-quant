#!/bin/bash
# Подтверждение emb=5 на 2.9B (закон 10: правка пресета требует обоих
# масштабов). ОДИН КОНФИГ НА ПРОЦЕСС -- закон 13.
#
#   bash tests/run_emb_split_2p9b.sh            # bf16, emb=6, emb=5, emb=4
#   bash tests/run_emb_split_2p9b.sh bf16 6 5   # только эти
#
# Результат копится в $RWKVQ_OUT между процессами; прерванный прогон
# продолжается с того же места -- уже посчитанные конфиги не
# пересчитываются, если не передавать их снова.
# Сводка в любой момент:
#   RWKVQ_OUT=/tmp/confirm_emb_2p9b.json \
#     ~/Develop/tests/venv/bin/python tests/confirm_emb_split.py --report
set -u
cd "$(dirname "$0")/.."

export RWKVQ_CKPT="${RWKVQ_CKPT:-$HOME/Develop/rwkv7-g1h-2.9b-ctx10240.pth}"
export RWKVQ_OUT="${RWKVQ_OUT:-/tmp/confirm_emb_2p9b.json}"
export RWKVQ_NSEQ="${RWKVQ_NSEQ:-16}"
export RWKVQ_SEQLEN="${RWKVQ_SEQLEN:-512}"
# batch_size=1 обязателен: модель занимает 5.9 ГБ на устройстве, логиты
# [2, 511, 65536] уже не влезают, и MPS роняет command buffer МОЛЧА,
# возвращая ppl=1.0000 вместо ошибки (защита -- в ablation.perplexity).
export RWKVQ_BATCH="${RWKVQ_BATCH:-1}"

PY="${PY:-$HOME/Develop/tests/venv/bin/python}"
VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(bf16 6 5 4)

# Шум Metal при ошибках командного буфера и предупреждение про act_stats
# отфильтрованы, но строка Insufficient Memory ОСТАВЛЕНА намеренно: это
# ровно тот случай, который нельзя пропустить.
FILTER='-e AGXG -e "label = " -e "device = " -e commandQueue -e retainedRef
        -e MallocStackLogging -e RuntimeWarning -e "ex2 = "'

for v in "${VARIANTS[@]}"; do
  echo "=================== конфиг $v"
  echo "   своп до:  $(sysctl -n vm.swapusage)"
  echo "   свободно: $(memory_pressure -Q 2>/dev/null | tail -1)"
  /usr/bin/time -l "$PY" -u tests/confirm_emb_split.py "$v" 2>&1 \
    | grep -v -e AGXG -e "label = " -e "device = " -e commandQueue \
              -e retainedRef -e MallocStackLogging -e RuntimeWarning \
              -e "ex2 = " -e "^\s*$" \
    | grep -v -e "voluntary" -e "involuntary" -e "reclaims" -e "faults" \
              -e "swaps" -e "messages" -e "signals" -e "switches" \
              -e "block input" -e "block output"
  echo "   своп после: $(sysctl -n vm.swapusage)"
  sleep 10
done

echo "=================== ГОТОВО"
RWKVQ_OUT="$RWKVQ_OUT" "$PY" tests/confirm_emb_split.py --report
