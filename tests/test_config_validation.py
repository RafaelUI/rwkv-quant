"""Гейт отказов QuantConfig / quantize() на конфигах, которые раньше молча делали не то, что написано (06.10, решение
владельца: отказ; находки разбора п. 1 проверки API):
  * неизвестное имя группы: QuantConfig(prooj=4) принимался, файл выходил целиком bf16;
  * group_scale_mode без group_scale: режим игнорировался, группа уходила в построчный RTN.
И исключение из отказа test_per_row_refusal: конфиг от calibrate() (построчные точки там измерены).

Свойства:
  V1 неизвестное имя в битах -> ValueError с этим именем и списком групп;
  V2 неизвестное имя в КАЖДОМ словарном поле (clip_percentiles, outlier_fracs, group_scale, group_scale_mode) -> ValueError;
  V3 все девять групп и псевдоним emb_head принимаются во всех полях; пресеты проходят validate();
  V4 режим без group_scale -> ValueError; с group_scale -- нет; псевдоним в одном поле и настоящая группа в другом -- нет;
  V5 правка ПОСЛЕ сборки (cfg.bits["prooj"], cfg.group_scale_mode без scale): quantize() отказывает до любой работы,
     файл не создан;
  V6 чтение манифестов: config_from_json на реальных файлах и на манифесте с незнакомой группой НЕ падает (strict=False);
  V7 среди кандидатов calibrate() (schema_space) нет построчных точек ниже 8 бит ни при какой ширине; конфиг с
     calibration_report и построчными 4 битами получает тот же отказ, что и любой другой (исключения нет).
Мутации (--mutate): проверка имён выключена (V1); имена проверяются только в bits (V2); проверка режимов выключена (V4);
quantize() не зовёт validate (V5); читалка строгая (V6); в кандидаты calibrate возвращён rtn@4 (V7).
    python tests/test_config_validation.py [--mutate]"""
import copy, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rwkv_quant import api, presets
from rwkv_quant.calibration.group_config import QuantConfig, GROUPS
from rwkv_quant.formats import codec, reader

CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
FILES = [os.path.expanduser("~/Develop/WKV-kvant/files_0110/" + n) for n in ("0p1b_compression.rwkvq", "0p1b_reduction.rwkvq", "0p4b_compression.rwkvq")]
FIELDS = ("clip_percentiles", "outlier_fracs", "group_scale", "group_scale_mode")


def raises(f, *frag):
    try: f()
    except ValueError as e: return all(s in str(e) for s in frag), str(e).split("\n")[0]
    except Exception as e: return False, "%s: %s" % (type(e).__name__, e)
    return False, "не упал"


def props(d):
    R = []
    def check(name, ok, info=""): R.append((name, bool(ok), str(info)[:200]))

    ok, info = raises(lambda: QuantConfig(prooj=4), "prooj", "proj", "cmix")
    check("V1 QuantConfig(prooj=4) -> отказ", ok, info)

    bad = []
    for f in FIELDS:
        kw = {f: {"prooj": 32 if f != "group_scale_mode" else "asym"}}
        if f == "group_scale_mode": kw["group_scale"] = {"proj": 32}
        ok, info = raises(lambda: QuantConfig(**kw), "prooj", f)
        if not ok: bad.append((f, info))
    check("V2 неизвестное имя в каждом словарном поле -> отказ", not bad, bad)

    try:
        for g in GROUPS + ["emb_head"]:
            QuantConfig(**{g: 8}, group_scale={g: 32}, group_scale_mode={g: "asym"}, clip_percentiles={g: 99.9}, outlier_fracs={g: 0.01})
        for p in presets.PRESETS.values(): p.validate(); copy.deepcopy(p).validate()
        check("V3 все группы, псевдоним и пресеты проходят", True)
    except Exception as e:
        check("V3 все группы, псевдоним и пресеты проходят", False, "%s: %s" % (type(e).__name__, e))

    ok, info = raises(lambda: QuantConfig(proj=4, group_scale_mode={"proj": "asym_sb6_aw"}), "group_scale", "proj")
    check("V4 режим без group_scale -> отказ", ok, info)
    try:
        QuantConfig(proj=4, group_scale={"proj": 32}, group_scale_mode={"proj": "asym_sb6_aw"})
        QuantConfig(emb_head=6, group_scale={"emb_head": 16}, group_scale_mode={"emb": "sym"})
        QuantConfig(emb=6, head=6, group_scale={"emb": 16, "head": 16}, group_scale_mode={"emb_head": "sym"})
        check("V4 режим с group_scale (в т.ч. через псевдоним) -> чисто", True)
    except Exception as e:
        check("V4 режим с group_scale (в т.ч. через псевдоним) -> чисто", False, e)

    f = os.path.join(d, "v5.rwkvq")
    c = copy.deepcopy(presets.REDUCTION); c.bits["prooj"] = 4
    ok, info = raises(lambda: api.quantize(CK, f, tokenizer=None, config=c, verbose=False), "prooj")
    c2 = copy.deepcopy(presets.REDUCTION); c2.group_scale_mode["g_lora"] = "asym"
    ok2, info2 = raises(lambda: api.quantize(CK, f, tokenizer=None, config=c2, verbose=False), "g_lora", "group_scale")
    check("V5 правка после сборки: quantize() отказывает до работы", ok and ok2 and not os.path.exists(f), (info, info2))

    try:
        n = 0
        for p in FILES:
            man, _ = codec.open_rwkvq(p); cfg = reader.config_from_json(man.get("config")); n += cfg is not None
        old = reader.config_from_json({"bits": {"proj": 4, "oldgroup": 8}, "group_scale_mode": {"proj": "asym_sb6"}})
        check("V6 манифесты читаются без отказов", n == len(FILES) and old.bits["oldgroup"] == 8, n)
    except Exception as e:
        check("V6 манифесты читаются без отказов", False, "%s: %s" % (type(e).__name__, e))

    # 06.10 (позже в тот же день): исключения для calibrate() больше нет -- он сам не выдаёт построчных точек ниже 8 бит
    from rwkv_quant.calibration import schema_space as S
    low = [(n, a, repr(c)) for n in (16, 64, 128, 768, 1024, 2048, 2560, 4096) for a in (True, False)
           for c in S.candidates_for(n, have_act_stats=True, all_2d=a) if c.family == "rtn" and c.bits < 8]
    cal = QuantConfig(w_lora=4); cal.calibration_report = {"groups": {"w_lora": {"chosen": "rtn@4"}}}
    okn, infon = raises(lambda: api.quantize(CK, os.path.join(d, "v7.rwkvq"), tokenizer=None, config=cal, verbose=False), "w_lora=4")
    check("V7 calibrate() не предлагает построчных точек ниже 8 бит; отчёт calibrate отказа не снимает", not low and okn, (low[:3], infon))
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
        Q = QuantConfig; init0 = Q.__init__
        def strict_init(self, *a, **k):
            k["strict"] = True; init0(self, *a, **k)
        muts = [("проверка имён выключена", lambda: setattr(Q, "_check_names", lambda self: None), "V1"),
                ("имена проверяются только в bits", lambda: setattr(Q, "_NAME_FIELDS", ("bits",)), "V2"),
                ("проверка режимов выключена", lambda: setattr(Q, "_check_modes", lambda self: None), "V4"),
                ("quantize() не зовёт validate", lambda: setattr(api, "_validate_config", lambda c: None), "V5"),
                ("читалка строгая", lambda: setattr(Q, "__init__", strict_init), "V6"),
                ("в кандидаты calibrate возвращён rtn@4", lambda: setattr(__import__("rwkv_quant.calibration.schema_space", fromlist=["x"]), "RTN_BITS", (8, 4)), "V7")]
        for name, apply, expect in muts:
            saved = (Q._check_names, Q._NAME_FIELDS, Q._check_modes, api._validate_config)
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                Q._check_names, Q._NAME_FIELDS, Q._check_modes, api._validate_config = saved
                __import__("rwkv_quant.calibration.schema_space", fromlist=["x"]).RTN_BITS = (8,)
                Q.__init__ = init0
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
