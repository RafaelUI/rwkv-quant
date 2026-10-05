"""Гейт варианта 2 (22.09): решение о быстрой норме записано в файл.

Проверяет на реальном 0.1B (сборка публичным quantize, секунды):
  1. preset="compression"                 -> манифест fast_ln=true, рантайм True (манифест)
  2. deepcopy(COMPRESSION)+поточечные биты -> то же (файл автоподбора -- compression-файл)
  3. preset="reduction"                   -> манифест false, рантайм False
  4. свой QuantConfig                     -> поля нет, рантайм решает умолчанием
  5. файл 20.09 (конфиг до emb 6)         -> True по истории compression
  6. явный fast_ln=False                  -> побеждает манифест
  7. load + save_rwkvq                    -> поле runtime сохраняется
  8. repr() пресетов не сдвинут полем     -> ключ preset_of и золота group_split цел
    python tests/test_fast_ln_intent.py
"""
import copy, json, os, struct, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rwkv_quant.api import quantize
from rwkv_quant.presets import COMPRESSION, REDUCTION
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.formats.writer import save_rwkvq
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

KV = "/Users/s/Develop/WKV-kvant"
CK = KV + "/rwkv7-g1d-0.1b.pth"
TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"
OLD = KV + "/compression_0p1b_l0_2009.rwkvq"
bad = []


def check(name, ok, info=""):
    print("  %s  %s %s" % ("ok  " if ok else "FAIL", name, info))
    if not ok:
        bad.append(name)


def manifest_runtime(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return json.loads(h["__metadata__"]["rwkvq"]).get("runtime")


def model(path, **kw):
    m = QuantRWKV7(load_raw(path), **kw)
    return m.fast_ln, m.fast_ln_source


REPR_C = ("QuantConfig(proj=4, w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=16, cmix=4, "
          "emb=6, head=5, overrides={'blocks.0.att.output.weight': 16, "
          "'blocks.0.tmix.o_proj.weight': 16, 'ffn.key.weight': 5, 'cmix.key.weight': 5})")
check("repr COMPRESSION не сдвинут полем", repr(COMPRESSION) == REPR_C, repr(COMPRESSION))
check("намерение пресетов", COMPRESSION.runtime_fast_ln is True and REDUCTION.runtime_fast_ln is False)

with tempfile.TemporaryDirectory() as d:
    f1 = os.path.join(d, "c.rwkvq")
    quantize(CK, f1, preset="compression", tokenizer=TOK, verbose=False, autopick=False, gptq=False)
    check("1 compression: манифест", manifest_runtime(f1) == {"fast_ln": True, "lora_q": True}, manifest_runtime(f1))
    check("1 compression: рантайм", model(f1) == (True, "манифест"), model(f1))

    cfg = copy.deepcopy(COMPRESSION)
    cfg.bits_overrides = dict({"blocks.3.att.value.weight": 6}, **cfg.bits_overrides)
    f2 = os.path.join(d, "r.rwkvq")
    quantize(CK, f2, config=cfg, tokenizer=TOK, verbose=False)
    check("2 автоподбор: манифест", manifest_runtime(f2) == {"fast_ln": True, "lora_q": True}, manifest_runtime(f2))
    check("2 автоподбор: рантайм", model(f2) == (True, "манифест"), model(f2))

    f3 = os.path.join(d, "red.rwkvq")
    quantize(CK, f3, preset="reduction", tokenizer=TOK, verbose=False, gptq=False)
    check("3 reduction: манифест", manifest_runtime(f3) == {"fast_ln": False, "lora_q": False}, manifest_runtime(f3))
    check("3 reduction: рантайм", model(f3) == (False, "манифест"), model(f3))

    own = QuantConfig(proj=6, cmix=6, emb_head=8, w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=16)
    f4 = os.path.join(d, "own.rwkvq")
    quantize(CK, f4, config=own, tokenizer=TOK, verbose=False, act_stats=None, allow_per_row=True)   # 06.10: построчно < 8 бит -- только явно
    check("4 свой конфиг: поля нет", manifest_runtime(f4) is None, manifest_runtime(f4))
    check("4 свой конфиг: умолчание", model(f4)[1] == "умолчание", model(f4))

    check("6 явный аргумент побеждает", model(f1, fast_ln=False) == (False, "аргумент"), model(f1, fast_ln=False))

    f7 = os.path.join(d, "rt.rwkvq")
    save_rwkvq(load_raw(f2), f7)
    check("7 пересохранение сохраняет runtime", manifest_runtime(f7) == {"fast_ln": True, "lora_q": True}, manifest_runtime(f7))

check("5 файл 20.09: история compression", model(OLD) == (True, "история compression"), model(OLD))
print("\nГЕЙТ ПРОЙДЕН" if not bad else "\nГЕЙТ КРАСНЫЙ: %s" % bad)
sys.exit(1 if bad else 0)
