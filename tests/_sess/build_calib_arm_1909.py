"""Сборка COMPRESSION с заданной статистикой AW (19.09) -- тем же публичным
путём, что у пользователя (api.quantize), только act_stats передан файлом.
    python build_calib_arm_1909.py <чекпоинт> <статистика.pt> <выход.rwkvq>"""
import sys
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
from rwkv_quant.api import quantize
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
quantize(sys.argv[1], sys.argv[3], preset="compression", tokenizer=TOK, act_stats=sys.argv[2], verbose=False)
print("готово:", sys.argv[3])
