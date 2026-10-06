"""Гейт отказов на неверное использование публичного API (06.10, решение владельца по находкам
tests/_sess/api_misuse_0610.py: 65 случаев на установленном колесе).

Было молча не то: output_path == чекпоинт (затирался); биты 12 / 4.0, clip_percentiles=250 (принимались); опечатка в
шаблоне bits_overrides (ничего не делала); act_stats не того формата (AW вырождался молча); gptq_calib с токенами за vocab
(на MPS -- мусор и записанный файл, на CUDA -- аппаратные ошибки); generate с id вне словаря (возвращал токены); step с
состоянием от другого батча (возвращал логиты). Было поздно или дорого: каталога выхода нет -- ошибка ПОСЛЕ всей работы;
gptq=True без sb6-матриц -- полный проход впустую. Было невнятно: не-RWKV чекпоинт, config словарём, device=0 и т. п.

Свойства (0.1B, из репозитория; tokenizer=None там, где отказ обязан прийти ДО сбора статистики):
  M1 каталога выхода нет -> FileNotFoundError до работы;
  M2 output_path == чекпоинт -> ValueError, файл чекпоинта не изменён (на КОПИИ чекпоинта);
  M3 типы аргументов: config словарём / строкой, device=0, act_stats=True -> TypeError с названием аргумента;
  M4 не-RWKV .pth и текстовый файл вместо чекпоинта -> ValueError с понятным текстом;
  M5 QuantConfig: биты 12 / 4.0 / "4" / 0 / True, clip 250, outlier 1.5, group_scale -32 -> ValueError; законные -- нет;
  M6 bits_overrides: опечатка -> ValueError с шаблоном; совпавший шаблон и шаблоны самих пресетов -- проходят;
  M7 act_stats: файл не того формата -> ValueError; настоящий файл статистики этого чекпоинта проверку проходит;
  M8 gptq_calib: токены за vocab -> ValueError; список строк -> TypeError;
  M9 gptq=True на конфиге без целей GPTQ -> предупреждение, файл записан быстро, записи gptq в манифесте нет;
     оба пресета (sb6 и sym) целями остаются;
  M10 токенизатор: путь к не-словарю -> TokenizerRequired; функция со строками -> TypeError;
  M11 рантайм: QuantRWKV7 от строки -> TypeError; generate с id >= vocab и < 0 -> ValueError, строкой -> TypeError;
      forward с одномерным idx -> ValueError, с state=None -> TypeError, с состоянием другого батча -> ValueError;
      обычный generate работает.
Мутации (--mutate): предполётная проверка аргументов выключена (M1); проверка значений QuantConfig выключена (M5);
проверка шаблонов выключена (M6); проверка act_stats выключена (M7); проверка промпта выключена (M11).
    python tests/test_api_misuse.py [--mutate]"""
import copy, hashlib, os, shutil, sys, tempfile, time, warnings
os.environ.setdefault("RWKVQ_DEVICE", "mps")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import mlx.core as mx
from rwkv_quant import api, presets
from rwkv_quant.calibration import act_stats as A
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats import codec
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.backends.metal import generate as G

WK = os.path.expanduser("~/Develop/WKV-kvant/")
CK = os.environ.get("CK", WK + "rwkv7-g1d-0.1b.pth")
TOK = os.environ.get("TOK", "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
FILE = WK + "files_0110/0p1b_reduction.rwkvq"
STATS = {}


def err(f, exc, *frag):
    try: f()
    except exc as e:
        return all(s in str(e) for s in frag), "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:110])
    except Exception as e:
        return False, "ДРУГОЕ %s: %s" % (type(e).__name__, str(e).split("\n")[0][:110])
    return False, "не упал"


def props(d):
    R = []
    def check(name, results):
        results = list(results); R.append((name, all(r[0] for r in results), [r[1] for r in results if not r[0]]))
    out = os.path.join(d, "o.rwkvq")
    q = lambda **kw: api.quantize(kw.pop("ck", CK), kw.pop("out", out), **dict(dict(tokenizer=None, verbose=False), **kw))

    check("M1 каталога выхода нет -> отказ до работы", [err(lambda: q(out=os.path.join(d, "нет", "x.rwkvq")), FileNotFoundError, "каталог")])

    cp = os.path.join(d, "copy.pth"); shutil.copyfile(CK, cp); size0 = os.path.getsize(cp)
    head0 = hashlib.md5(open(cp, "rb").read(1 << 20)).hexdigest()
    r = err(lambda: q(ck=cp, out=cp, tokenizer=TOK, preset="reduction", gptq=False), ValueError, "совпадает")
    same = os.path.getsize(cp) == size0 and hashlib.md5(open(cp, "rb").read(1 << 20)).hexdigest() == head0
    check("M2 output_path == чекпоинт -> отказ, чекпоинт цел", [r, (same, "чекпоинт ИЗМЕНЁН")])

    check("M3 типы аргументов", [err(lambda: q(config={"proj": 4}), TypeError, "config"), err(lambda: q(config="compression"), TypeError, "config"),
                                 err(lambda: q(device=0), TypeError, "device"), err(lambda: q(act_stats=True), TypeError, "act_stats")])

    junk = os.path.join(d, "junk.pth"); torch.save({"a.weight": torch.randn(8, 8)}, junk)
    check("M4 не чекпоинт RWKV-7", [err(lambda: q(ck=junk), ValueError, "не похож"), err(lambda: q(ck=TOK), ValueError, "не удалось прочитать")])

    gs = dict(group_scale={"proj": 32}, group_scale_mode={"proj": "asym_sb6"})
    bad = [("биты 12", lambda: QuantConfig(proj=12)), ("биты 4.0", lambda: QuantConfig(proj=4.0, **gs)), ("биты '4'", lambda: QuantConfig(proj="4", **gs)),
           ("биты 0", lambda: QuantConfig(proj=0)), ("биты True", lambda: QuantConfig(proj=True)), ("clip 250", lambda: QuantConfig(proj=8, clip_percentiles={"proj": 250})),
           ("outlier 1.5", lambda: QuantConfig(proj=8, outlier_fracs={"proj": 1.5})), ("gs -32", lambda: QuantConfig(proj=4, group_scale={"proj": -32}, group_scale_mode={"proj": "asym_sb6"}))]
    res = [(lambda r, n: (r[0], n + ": " + r[1]))(err(f, ValueError, "недопустим"), n) for n, f in bad]
    try:
        for b in (1, 2, 4, 5, 6, 7, 8, 16): QuantConfig(proj=b, small=b)
        QuantConfig(proj=8, clip_percentiles={"proj": 99.9}, outlier_fracs={"proj": 0.02}); QuantConfig(proj=4, **gs)
        c = copy.deepcopy(presets.COMPRESSION); c.bits_overrides = dict({"blocks.0.att.key.weight": 5}, **c.bits_overrides); c.validate()
        for p in presets.PRESETS.values(): p.validate()
        res.append((True, ""))
    except Exception as e:
        res.append((False, "законное значение отклонено: %s" % e))
    check("M5 значения QuantConfig", res)

    c1 = copy.deepcopy(presets.COMPRESSION); c1.bits_overrides = dict({"blocks.0.att.kee.weight": 5}, **c1.bits_overrides)
    c2 = copy.deepcopy(presets.COMPRESSION); c2.bits_overrides = dict({"blocks.0.att.key.weight": 5}, **c2.bits_overrides)
    ok_pass = []
    for name, c in (("совпавший шаблон", c2), ("compression", presets.COMPRESSION), ("reduction", presets.REDUCTION)):
        try: api._refuse_unmatched_overrides(CK, c); ok_pass.append((True, ""))
        except Exception as e: ok_pass.append((False, "%s отклонён: %s" % (name, str(e)[:80])))
    check("M6 шаблоны bits_overrides", [err(lambda: q(config=c1), ValueError, "kee")] + ok_pass)

    if "path" not in STATS:
        cache = tempfile.mkdtemp(prefix="rq_misuse_stats_"); STATS["dir"] = cache
        c0 = A.CACHE_DIR; A.CACHE_DIR = cache
        try: _, sig = A.collect(CK, TOK, verbose=False)
        finally: A.CACHE_DIR = c0
        STATS["path"] = os.path.join(cache, "act_%s.pt" % sig)
    badst = os.path.join(d, "bad_act.pt"); torch.save({"x": torch.zeros(3)}, badst)
    try: api._check_act_stats(STATS["path"], CK, presets.COMPRESSION); real = (True, "")
    except Exception as e: real = (False, "настоящая статистика отклонена: %s" % str(e)[:120])
    check("M7 act_stats не того формата -> отказ; настоящий проходит",
          [err(lambda: q(preset="compression", act_stats=badst, gptq=False, autopick=False), ValueError, "не подходит"), real])

    g = dict(preset="compression", act_stats=STATS["path"], gptq=True, autopick=False)
    check("M8 gptq_calib", [err(lambda: q(gptq_calib=torch.full((600, 64), 70000), **g), ValueError, "вне словаря"),
                            err(lambda: q(gptq_calib=["текст"] * 600, **g), TypeError, "gptq_calib")])

    f9 = os.path.join(d, "g9.rwkvq"); t0 = time.time()
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always"); q(out=f9, config=QuantConfig(proj=8), gptq=True)
        man = codec.open_rwkvq(f9)[0]
        r9 = (any("gptq=True пропущен" in str(w.message) for w in ws) and not man.get("gptq") and time.time() - t0 < 120,
              "предупреждений %d, gptq в манифесте %r, %.0f с" % (len(ws), bool(man.get("gptq")), time.time() - t0))
    except Exception as e:
        r9 = (False, "%s: %s" % (type(e).__name__, str(e)[:100]))
    # оба пресета ОБЯЗАНЫ остаться целями GPTQ (reduction -- sym_aw): первая редакция проверки пропускала reduction
    tg = (api._gptq_has_targets(presets.REDUCTION) and api._gptq_has_targets(presets.COMPRESSION), "пресет перестал быть целью GPTQ")
    check("M9 gptq=True без целей -> предупреждение и быстрый пропуск; пресеты -- цели", [r9, tg])

    A._TOK_MEMO.clear()
    check("M10 токенизатор", [err(lambda: A._encoder(A.CORPUS), A.TokenizerRequired, "не словарь"),
                              err(lambda: A.calib_windows(lambda s: s.split()), TypeError, "целых id")])
    A._TOK_MEMO.clear()

    m = qm.QuantRWKV7(load_raw(FILE)); V = m.vocab_size
    try:
        toks, _ = G.generate(m, [5, 6, 7], 4); okg = (len(toks) == 4 and all(0 <= t < V for t in toks), "generate вернул %r" % (toks,))
    except Exception as e:
        okg = (False, "обычный generate упал: %s" % e)
    check("M11 рантайм Metal", [err(lambda: qm.QuantRWKV7(FILE), TypeError, "load_raw"),
                                err(lambda: G.generate(m, [V + 5, 5], 2), ValueError, "вне словаря"), err(lambda: G.generate(m, [-1, 5], 2), ValueError, "вне словаря"),
                                err(lambda: G.generate(m, "привет", 2), TypeError, "токенизируйте"),
                                err(lambda: m.forward_stateful(mx.array([5, 6, 7]), m.init_state(1)), ValueError, "[B, T]"),
                                err(lambda: m.forward_stateful(mx.array([[5, 6]]), None), TypeError, "init_state"),
                                err(lambda: m.forward_stateful(mx.array([[5, 6]]), m.init_state(2)), ValueError, "батч"), okg])
    return R


def run(label):
    with tempfile.TemporaryDirectory(prefix="rq_misuse_gate_") as d:
        R = props(d)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + "; ".join(info)[:300]))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    try:
        ok = not run("КОНТРОЛЬ")
        if "--mutate" in sys.argv:
            Q = QuantConfig
            muts = [("предполётная проверка аргументов выключена", lambda: setattr(api, "_preflight_args", lambda *a: None), "M1"),
                    ("проверка значений QuantConfig выключена", lambda: setattr(Q, "_check_values", lambda self: None), "M5"),
                    ("проверка шаблонов выключена", lambda: setattr(api, "_refuse_unmatched_overrides", lambda *a: None), "M6"),
                    ("проверка act_stats выключена", lambda: setattr(api, "_check_act_stats", lambda *a: None), "M7"),
                    ("проверка промпта выключена", lambda: setattr(G, "_check_prompt", lambda *a: None), "M11")]
            for name, apply, expect in muts:
                saved = (api._preflight_args, Q._check_values, api._refuse_unmatched_overrides, api._check_act_stats, G._check_prompt)
                apply()
                try:
                    failed = run("МУТАЦИЯ «%s»" % name)
                except Exception as e:
                    failed = ["%s (прогон упал: %s: %s)" % (expect, type(e).__name__, str(e)[:80])]; print("  мутация уронила прогон:", failed[0])
                finally:
                    api._preflight_args, Q._check_values, api._refuse_unmatched_overrides, api._check_act_stats, G._check_prompt = saved
                caught = any(f.startswith(expect) for f in failed)
                print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect)); ok &= caught
            ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    finally:
        if STATS.get("dir"): shutil.rmtree(STATS["dir"], ignore_errors=True)   # свой временный каталог гейта
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
