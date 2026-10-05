"""Гейт отказа quantize() на построчном RTN ниже 8 бит (06.10, находка 1 проверки API от 04.10).

Было: quantize(ckpt, out, tokenizer=tok, config=QuantConfig(proj=4, cmix=4, emb=6, head=6)) -- пример из docstring --
молча писал файл с KL 3.10 (0.1B): группы без group_scale идут в построчный RTN. Решение владельца -- отказ.

Свойства (0.1B, из репозитория):
  P1 пример из прежнего docstring -> ValueError, названы все четыре группы, файл не создан;
  P2 каждая из девяти групп построчно при 4 и 6 битах -- нарушитель, при 8 и 16 -- нет (граница ровно на 8);
  P3 bits_overrides: 4 бита точечно на матрицу группы БЕЗ group_scale -- нарушитель (ровно эти ключи);
     точечно на матрицу группы С group_scale -- нет;
  P4 оба пресета и конфиг от пресета с правкой bits -- нарушителей нет (у пресетов g_lora=8 построчно -- законно);
  P5 allow_per_row=True пишет файл, побайтно равный прямому quantize_file (флаг не меняет байты);
  P6 отказ идёт ДО любой работы: с tokenizer=None и AW-режимом у другой группы приходит ValueError отказа,
     а не TokenizerRequired сбора статистики.
Мутации (--mutate; каждая обязана быть пойманной): порог 8 -> 5 (ловит P2); порог 8 -> 9 (P4: пресеты под отказом);
предполётная проверка не видит bits_overrides (P3); отказ выключен (P1).
    python tests/test_per_row_refusal.py [--mutate]"""
import copy, hashlib, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rwkv_quant import api, presets
from rwkv_quant.calibration.group_config import QuantConfig, GROUPS
from rwkv_quant.calibration.act_stats import TokenizerRequired
from rwkv_quant.formats import writer as W

CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()


def props(d):
    """-> список (имя свойства, ок, подробность)."""
    R = []
    def check(name, ok, info=""): R.append((name, bool(ok), str(info)[:200]))

    # P1
    f = os.path.join(d, "p1.rwkvq")
    try:
        api.quantize(CK, f, tokenizer=None, config=QuantConfig(proj=4, cmix=4, emb=6, head=6), verbose=False)
        check("P1 пример docstring -> отказ", False, "не упал")
    except ValueError as e:
        msg = str(e)
        check("P1 пример docstring -> отказ", all(s in msg for s in ("proj=4", "cmix=4", "emb=6", "head=6", "allow_per_row")), msg.split("\n")[0])
    except Exception as e:
        check("P1 пример docstring -> отказ", False, "%s: %s" % (type(e).__name__, e))
    check("P1 файл не создан", not os.path.exists(f))

    # P2
    bad = []
    for g in GROUPS:
        for b, want in ((4, True), (6, True), (7, True), (8, False), (16, False)):
            got = W.per_row_low_bits(CK, QuantConfig(**{g: b}))
            if bool(got) != want or any(x[1] != g or x[2] != b for x in got):
                bad.append((g, b, len(got)))
    check("P2 граница ровно на 8 битах, все девять групп", not bad, bad)

    # P3
    c = copy.deepcopy(presets.REDUCTION); c.bits_overrides = dict({"att.g1": 4}, **c.bits_overrides)
    got = W.per_row_low_bits(CK, c)
    check("P3 override 4 бита на g_lora (без group_scale) -> нарушитель",
          bool(got) and all(k.endswith("att.g1") and g == "g_lora" and b == 4 for k, g, b in got), got[:2])
    c = copy.deepcopy(presets.REDUCTION); c.bits_overrides = dict({"blocks.3.att.key.weight": 6}, **c.bits_overrides)
    check("P3 override на proj (с group_scale) -> чисто", W.per_row_low_bits(CK, c) == [], W.per_row_low_bits(CK, c)[:2])

    # P4
    for name, p in presets.PRESETS.items():
        got = W.per_row_low_bits(CK, p)
        check("P4 пресет %s чист" % name, got == [], got[:2])
    c = copy.deepcopy(presets.COMPRESSION); c.bits["proj"] = 5
    check("P4 конфиг от пресета (proj=5) чист", W.per_row_low_bits(CK, c) == [])

    # P5
    cfg = QuantConfig(proj=6)
    fa, fb = os.path.join(d, "p5a.rwkvq"), os.path.join(d, "p5b.rwkvq")
    try:
        api.quantize(CK, fa, tokenizer=None, config=cfg, verbose=False, allow_per_row=True)
        W.quantize_file(CK, fb, copy.deepcopy(cfg), verbose=False)
        check("P5 allow_per_row=True == прямой quantize_file побайтно", md5(fa) == md5(fb), (md5(fa), md5(fb)))
    except Exception as e:
        check("P5 allow_per_row=True == прямой quantize_file побайтно", False, "%s: %s" % (type(e).__name__, e))

    # P6
    c = QuantConfig(proj=4, cmix=4, group_scale={"cmix": 32}, group_scale_mode={"cmix": "asym_sb6_aw"})
    try:
        api.quantize(CK, os.path.join(d, "p6.rwkvq"), tokenizer=None, config=c, verbose=False)
        check("P6 отказ до сбора статистики", False, "не упал")
    except TokenizerRequired:
        check("P6 отказ до сбора статистики", False, "сначала дошло до токенизатора")
    except ValueError as e:
        check("P6 отказ до сбора статистики", "proj=4" in str(e) and "cmix" not in str(e).split("\n")[0], str(e).split("\n")[0])
    return R


def run(label):
    with tempfile.TemporaryDirectory() as d:
        R = props(d)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        muts = [("порог 8 -> 5", lambda: setattr(W, "PER_ROW_MIN_BITS", 5), "P2"),
                ("порог 8 -> 9", lambda: setattr(W, "PER_ROW_MIN_BITS", 9), "P4"),
                ("проверка не видит bits_overrides", lambda: setattr(W, "_bits_of", lambda key, group, cfg: cfg.bits[group]), "P3"),
                ("отказ выключен", lambda: setattr(api, "_refuse_per_row", lambda *a, **k: None), "P1")]
        for name, apply, expect in muts:
            saved = (W.PER_ROW_MIN_BITS, W._bits_of, api._refuse_per_row)
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                W.PER_ROW_MIN_BITS, W._bits_of, api._refuse_per_row = saved
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
