"""Гейт отказа autopick на раскладке не sb6 (06.10, решение владельца; находка проверки «неверным использованием»).

Было: quantize(..., preset="reduction", autopick=True) проходил измерение (час на 1.5B) и падал в писателе
NotImplementedError «real_gw: mode=sym_aw bits=7»: autopick назначает биты по лестнице sb6 (4 / 5 / 6 -> bf16).

Свойства (0.1B):
  A1 preset="reduction" + autopick=True -> ValueError про autopick ДО любой работы (tokenizer=None: если бы дошло до
     статистики, пришёл бы TokenizerRequired), файл не создан; истинное значение любого типа ("да", 1) -- то же;
  A2 свой построчный конфиг (proj=8) + autopick=True -> тот же отказ;
  A3 COMPRESSION и конфиг от него с правкой битов в пределах 4..6 -- поддерживаются (пусто); REDUCTION -- нет, названы
     его группы; группа в bf16 проверке не мешает;
  A4 без autopick (None / False) reduction не задет: preset="reduction", autopick=None, gptq=False пишет файл.
Мутации (--mutate): проверка выключена (A1); в поддерживаемые биты добавлено 8 и режим не проверяется (A3).
    python tests/test_autopick_refusal.py [--mutate]"""
import copy, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rwkv_quant import api, presets
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.act_stats import TokenizerRequired

CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
TOK = os.environ.get("TOK", "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")


def refused(f):
    try: f()
    except ValueError as e: return "autopick" in str(e), "ValueError: " + str(e)[:120]
    except TokenizerRequired: return False, "дошло до токенизатора (TokenizerRequired)"
    except Exception as e: return False, "%s: %s" % (type(e).__name__, str(e)[:120])
    return False, "не упал"


def props(d):
    R = []
    def check(name, ok, info=""): R.append((name, bool(ok), str(info)[:220]))
    f = os.path.join(d, "a.rwkvq")
    r = [refused(lambda v=v: api.quantize(CK, f, preset="reduction", tokenizer=None, autopick=v, verbose=False)) for v in (True, "да", 1)]
    check("A1 reduction + autopick=True -> отказ до работы", all(x[0] for x in r) and not os.path.exists(f), [x[1] for x in r])
    r2 = refused(lambda: api.quantize(CK, f, config=QuantConfig(proj=8), tokenizer=None, autopick=True, verbose=False))
    check("A2 построчный конфиг + autopick=True -> отказ", r2[0] and not os.path.exists(f), r2[1])
    c5 = copy.deepcopy(presets.COMPRESSION); c5.bits["proj"] = 5
    cb = copy.deepcopy(presets.COMPRESSION); cb.bits["cmix"] = 16
    red = api._autopick_unsupported(presets.REDUCTION)
    check("A3 COMPRESSION и его правки -- поддерживаются, REDUCTION -- нет",
          api._autopick_unsupported(presets.COMPRESSION) == [] and api._autopick_unsupported(c5) == [] and api._autopick_unsupported(cb) == []
          and {g for g, _, _ in red} >= {"proj", "head"}, (api._autopick_unsupported(presets.COMPRESSION), red))
    f4 = os.path.join(d, "r.rwkvq")
    try:
        api.quantize(CK, f4, preset="reduction", tokenizer=TOK, gptq=False, verbose=False)
        api.quantize(CK, f4 + "2", preset="reduction", tokenizer=TOK, gptq=False, autopick=False, verbose=False)
        check("A4 reduction без autopick не задет", os.path.exists(f4) and os.path.exists(f4 + "2"))
    except Exception as e:
        check("A4 reduction без autopick не задет", False, "%s: %s" % (type(e).__name__, str(e)[:150]))
    return R


def run(label):
    with tempfile.TemporaryDirectory(prefix="rq_ap_gate_") as d:
        R = props(d)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        def m_off(): api._refuse_autopick = lambda c: None
        def m_wide(): api._autopick_unsupported = lambda c: [(g, None, c.bits[g]) for g in api._AUTOPICK_GROUPS if c.bits[g] < 4]
        for name, apply, expect in (("проверка выключена", m_off, "A1"), ("принимаются любые режим и биты от 4", m_wide, "A3")):
            saved = (api._refuse_autopick, api._autopick_unsupported); apply()
            try: failed = run("МУТАЦИЯ «%s»" % name)
            finally: api._refuse_autopick, api._autopick_unsupported = saved
            caught = any(x.startswith(expect) for x in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect)); ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
