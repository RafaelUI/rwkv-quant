"""Побайтовое сравнение тензоров двух .rwkvq (контроль детерминизма сборки)."""
import sys
from safetensors import safe_open
a, b = sys.argv[1], sys.argv[2]
fa, fb = safe_open(a, "pt"), safe_open(b, "pt")
ka, kb = set(fa.keys()), set(fb.keys())
print("ключи: %d / %d, общих %d" % (len(ka), len(kb), len(ka & kb)))
diff = [k for k in sorted(ka & kb) if not __import__("torch").equal(fa.get_tensor(k), fb.get_tensor(k))]
print("различающихся тензоров: %d %s" % (len(diff), diff[:5]))
