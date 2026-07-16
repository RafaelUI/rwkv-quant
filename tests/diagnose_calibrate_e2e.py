"""Открытый вопрос №2: combined_sanity_check() ранее тестировался только
напрямую (test_combined_sanity_check.py, с РУЧНО заданным best_bits), не
через полный calibrate() end-to-end. Прогоняем calibrate() целиком на 1.5B
и проверяем, что итоговый конфиг разумен и repair-loop не зацикливается."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rwkv_quant.api import calibrate

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")

t0 = time.time()
config = calibrate(CKPT_PTH, CORPUS, device="mps", verbose=True)
print(f"\nTotal time: {time.time()-t0:.1f}s")
print(f"Final config: {config}")
