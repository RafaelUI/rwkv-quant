"""Гейт копии токенизатора (06.10, находка 4 проверки API): rwkv_quant/world_tokenizer.py -- копия
rwkv_metal/tokenizer/world_tokenizer.py для машин, где rwkv-metal не импортируется (Linux). Вторая копия чужой
реализации допустима только под гейтом тождественности (закон 23).

Свойства (на Mac, где есть оба):
  T1 текст копии после заголовка == тексту оригинала после его заголовка;
  T2 токены копии == токенам оригинала на всех чанках калибровочного корпуса; decode(encode(x)) == x;
  T3 без rwkv_metal (импорт заблокирован) act_stats._encoder(путь) работает и даёт те же токены; несуществующий путь --
     TokenizerRequired, а не FileNotFoundError из недр;
  T4 pyproject: mlx и rwkv-metal -- только под маркером darwin, numpy / torch / safetensors -- без маркера.
Мутации (--mutate): копия теряет последний токен (T2); копия недоступна (T3); маркер снят с mlx (T4).
    python tests/test_world_tokenizer_copy.py [--mutate]"""
import importlib, os, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from rwkv_quant import world_tokenizer as WQ
from rwkv_quant.calibration import act_stats as A
import rwkv_metal.tokenizer.world_tokenizer as WM

TOK = os.environ.get("TOK", os.path.join(os.path.dirname(WM.__file__), "rwkv_vocab_v20230424.txt"))
CHUNKS = [c.strip() for c in re.split(r"—+ CHUNK —+", open(A.CORPUS, encoding="utf-8").read()) if c.strip()]
PYPROJECT = [open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()]


def body(path):
    s = open(path, encoding="utf-8").read()
    assert s.startswith('"""'), path
    return s[s.index('"""', 3) + 3:]


def props():
    R = []
    def check(name, ok, info=""): R.append((name, bool(ok), str(info)[:200]))
    check("T1 текст копии == оригиналу", body(WQ.__file__) == body(WM.__file__))

    q, m = WQ.WorldTokenizer(TOK), WM.WorldTokenizer(TOK)
    bad = [i for i, c in enumerate(CHUNKS) if q.encode(c) != m.encode(c)]
    rt = all(q.decode(q.encode(c)) == c for c in CHUNKS[:8])
    check("T2 токены копии == оригиналу на корпусе (%d чанков), decode обратим" % len(CHUNKS), not bad and rt, (bad[:5], rt))

    saved = {k: v for k, v in sys.modules.items() if k == "rwkv_metal" or k.startswith("rwkv_metal.")}
    try:
        for k in saved: sys.modules[k] = None                      # import rwkv_metal... -> ImportError
        enc = A._encoder(TOK)
        same = all(enc(c) == m.encode(c) for c in CHUNKS[:8]); mod = type(enc.__self__).__module__
        try:
            A._encoder("/нет/такого/словаря.txt"); miss = "не упал"
        except A.TokenizerRequired: miss = "TokenizerRequired"
        except Exception as e: miss = type(e).__name__
        check("T3 без rwkv_metal путь-токенизатор работает (копия), нет файла -- TokenizerRequired",
              same and mod == "rwkv_quant.world_tokenizer" and miss == "TokenizerRequired", (same, mod, miss))
    except Exception as e:
        check("T3 без rwkv_metal путь-токенизатор работает (копия), нет файла -- TokenizerRequired", False, "%s: %s" % (type(e).__name__, e))
    finally:
        sys.modules.update(saved)

    deps = re.search(r"^dependencies = \[(.*?)^\]", PYPROJECT[0], re.S | re.M).group(1)
    deps = [l.strip().strip(",").strip('"') for l in deps.split("\n") if l.strip().startswith('"')]
    mac = [d for d in deps if "sys_platform == 'darwin'" in d]; rest = [d for d in deps if d not in mac]
    check("T4 pyproject: mlx и rwkv-metal только под darwin", sorted(d.split(">")[0].split(";")[0].strip() for d in mac) == ["mlx", "rwkv-metal"]
          and sorted(rest) == ["numpy", "safetensors", "torch"], deps)
    return R


def run(label):
    R = props(); failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        enc0 = WQ.WorldTokenizer.encode; py0 = PYPROJECT[0]
        def m_copy(): WQ.WorldTokenizer.encode = lambda self, s: enc0(self, s)[:-1]
        def m_gone(): sys.modules["rwkv_quant.world_tokenizer"] = None
        def m_mark(): PYPROJECT[0] = py0.replace("\"mlx>=0.31; sys_platform == 'darwin'\"", '"mlx>=0.31"')
        for name, apply, expect in (("копия теряет последний токен", m_copy, "T2"), ("копия недоступна", m_gone, "T3"), ("маркер снят с mlx", m_mark, "T4")):
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                WQ.WorldTokenizer.encode = enc0; sys.modules["rwkv_quant.world_tokenizer"] = WQ; PYPROJECT[0] = py0
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect)); ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
