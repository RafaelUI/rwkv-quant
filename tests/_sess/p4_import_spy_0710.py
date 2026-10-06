"""07.10: кто пробует импортировать mlx / rwkv_metal в пути quantize() на Linux (п. 4).
Шпион на sys.meta_path записывает КАЖДУЮ попытку найти модуль mlx* / rwkv_metal* со стеком -- независимо от того,
установлен ли пакет. Запускать на установленном колесе (PYTHONPATH=<site>, cwd=/tmp), с чистым HOME (пустой кеш)
и повторно с тёплым.
    python p4_import_spy_0710.py <ckpt> <tok> <device> <out.rwkvq> <preset> [gptq:0|1]"""
import importlib.abc, json, os, sys, time, traceback, warnings
LOG = []
class Spy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("mlx", "rwkv_metal"):
            LOG.append((name, [("%s:%d %s" % (f.filename.split("site/")[-1], f.lineno, f.name)) for f in traceback.extract_stack()[:-1]
                               if "importlib" not in f.filename][-6:]))
        return None
sys.meta_path.insert(0, Spy())
CK, TOK, DEV, OUT, PRESET = sys.argv[1:6]
GPTQ = None if len(sys.argv) < 7 else bool(int(sys.argv[6]))
import rwkv_quant
print("пакет:", rwkv_quant.__file__, "| HOME", os.environ.get("HOME"), "| cwd", os.getcwd())
t0 = time.time()
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    rwkv_quant.quantize(CK, OUT, preset=PRESET, tokenizer=TOK, device=DEV, gptq=GPTQ, verbose=False)
print("quantize: %.0f с, файл %.2f МБ" % (time.time() - t0, os.path.getsize(OUT) / 1e6))
print("предупреждения:", [str(w.message)[:160] for w in W])
print("в sys.modules:", sorted(m for m in sys.modules if m.split(".")[0] in ("mlx", "rwkv_metal")))
print("загружено из пакета:", sorted(m for m in sys.modules if m.startswith("rwkv_quant"))) 
print("попыток импорта mlx / rwkv_metal: %d" % len(LOG))
for name, st in LOG:
    print("  *", name)
    for s in st: print("       ", s)
