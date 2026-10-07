"""Гейт единой таблицы поддержки реального писателя и предполётной проверки quantize() (07.10, п. 3 сессии).

Было: условия «какой режим на каких битах и при какой ширине пишется в .rwkvq» жили в семи местах; неподдержанная
пара падала из писателя ПОСРЕДИ работы (NotImplementedError, AssertionError, «too many values to unpack»), любой режим
на «sym» ("sym_typo") молча считался sym, неизвестный режим на измерительном пути молча считался asym.
Стало: одна таблица groupwise.REAL_GW_SUPPORT + real_gw_refusal; ею выбирает ветку писатель, по ней же
writer.unsupported_gw и отказ quantize() до работы; QuantConfig отказывает на неизвестном режиме.

Свойства (0.1B, из репозитория; тензоры настоящие, по одному на каждую форму группы; у emb / head OUT обрезан до
256 строк -- они не LoRA, писатель их не транспонирует, IN не меняется):
  P1 таблица == писатель: по сетке режим x биты 1..8 x блок {16, 32, 64, 33} x тензор «причины отказа нет» <=>
     quantize_tensor(real_gw=True) отработал;
  P2 якоря, записанные в гейте, а не взятые из таблицы: sb6 -- ровно 4/5/6, sym -- ровно 6/8, asym -- ровно 5..8,
     mxfp4 -- ничего; sb6 с блоком 64 на ширине 768 -- отказ; asym с блоком 33 -- пишется;
  P3 LoRA: форма для проверки -- ПОСЛЕ транспонирования: w1 [768, 64] в sb6 пишется, w2 [64, 768] -- нет;
     unsupported_gw называет ровно те ключи, на которых падает писатель;
  P4 оба пресета чисты (unsupported_gw == []) на каждом чекпоинте из CKS;
  P5 quantize() отказывает ValueError ДО работы (файла нет; с tokenizer=None и AW у другой группы -- не
     TokenizerRequired): sb6 на 7 битах, sb6 с блоком 64, group_scale у small (не 2-D);
  P6 real_gw=False остаётся шире: тот же sb6 на 7 битах пишет файл;
  P7 QuantConfig: все режимы GW_MODES принимаются; "bogus" и "sym_typo" -- ValueError; strict=False пропускает;
  P8 "sym_typo" мимо проверки (strict=False) писатель НЕ принимает за sym: NotImplementedError, и unsupported_gw его называет.
Мутации (--mutate; каждая обязана быть пойманной): sb6 пишет 7 бит (P2); "sym_typo" внесён в таблицу (P8);
"bogus" внесён в GW_MODES (P7); суперблок sb6 8 -> 4 (P1); транспонирование выключено (P3); отказ в api выключен (P5).
    python tests/test_gw_support.py [--mutate]      CKS="a.pth:b.pth" -- доп. чекпоинты для P4"""
import collections, copy, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from rwkv_quant import api, presets
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.calibration.act_stats import TokenizerRequired
from rwkv_quant.calibration import groupwise as G
from rwkv_quant.formats import writer as W

CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
CKS = [CK] + [os.path.expanduser(p) for p in os.environ.get("CKS", "").split(":") if p]
SD = torch.load(CK, map_location="cpu", mmap=True)
REPS = collections.OrderedDict()
for _k, _w in SD.items():
    _g = W._match_group(_k)
    if W._is_quantized(_k, _g, _w.dim()):
        REPS.setdefault((_g, tuple(_w.shape)), _k)
GS = (16, 32, 64, 33)
PROJ = next(k for (g, sh), k in REPS.items() if g == "proj" and sh == (768, 768))


def cfg_for(g, mode, gs, b):
    return QuantConfig(group_scale={g: gs}, group_scale_mode=({g: mode} if mode else {}), strict=False, **{g: b})


def writes(k, w, c):
    try:
        W.quantize_tensor(k, w, c, real_gw=True); return True, ""
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, str(e)[:60])


def props(d):
    R = []
    def check(name, ok, info=""): R.append((name, bool(ok), str(info)[:260]))

    # P1
    bad, n = [], 0
    for (g, sh), k in REPS.items():
        w = SD[k].float()
        if g in ("emb", "head") and w.shape[0] > 256: w = w[:256].contiguous()
        shape = tuple(w.shape)[::-1] if W._transposes(k, w.dim()) else tuple(w.shape)
        for mode in [None] + list(G.GW_MODES):
            for gs in GS:
                for b in range(1, 9):
                    n += 1
                    why = G.real_gw_refusal(mode or "asym", b, gs, shape)
                    ok, err = writes(k, w, cfg_for(g, mode, gs, b))
                    if (why is None) != ok: bad.append((k.split(".")[-1], mode, gs, b, why, err))
    check("P1 таблица == писатель на сетке (%d точек)" % n, not bad and n > 2000, "%d расхождений, напр. %s" % (len(bad), bad[:2]))

    # P2
    want = {"asym_sb6": {4, 5, 6}, "asym_sb6_search": {4, 5, 6}, "asym_sb6_aw": {4, 5, 6}, "sym": {6, 8}, "sym_plain": {6, 8},
            "sym_aw": {6, 8}, "asym": {5, 6, 7, 8}, "mxfp4": set()}
    got = {m: {b for b in range(1, 9) if G.real_gw_refusal(m, b, 32, (768, 768)) is None} for m in want}
    check("P2 якоря режим -> биты", got == want and set(want) == set(G.GW_MODES), {m: sorted(got[m]) for m in got if got[m] != want[m]})
    check("P2 sb6 с блоком 64 на ширине 768 -- отказ", G.real_gw_refusal("asym_sb6", 6, 64, (768, 768)) is not None)
    check("P2 asym с блоком 33 -- пишется", G.real_gw_refusal("asym", 6, 33, (768, 768)) is None)

    # P3
    c = QuantConfig(w_lora=6, group_scale={"w_lora": 32}, group_scale_mode={"w_lora": "asym_sb6"})
    rep = {k for k, *_ in W.unsupported_gw(CK, c)}
    act = {k for k in SD if W._match_group(k) == "w_lora" and W._is_quantized(k, "w_lora", SD[k].dim())
           and not writes(k, SD[k].float(), c)[0]}
    check("P3 unsupported_gw == ключи, на которых падает писатель", rep == act, (sorted(rep ^ act)[:3], len(rep), len(act)))
    check("P3 w1 [768, 64] пишется, w2 [64, 768] -- нет (форма после транспонирования)",
          rep and all(k.endswith("w2") for k in rep) and tuple(SD[next(iter(rep))].shape) == (64, 768) if rep else False, sorted(rep)[:2])

    # P4
    for ck in CKS:
        for name, p in presets.PRESETS.items():
            got = W.unsupported_gw(ck, p)
            check("P4 пресет %s чист на %s" % (name, os.path.basename(ck)), got == [], got[:2])

    # P5
    cases = [("sb6 на 7 битах", QuantConfig(proj=7, cmix=4, group_scale={"proj": 32, "cmix": 32},
                                             group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6_aw"}), ("proj", "4/5/6")),
             ("sb6 с блоком 64", QuantConfig(proj=6, group_scale={"proj": 64}, group_scale_mode={"proj": "asym_sb6"}), ("proj", "512")),
             ("group_scale у small", QuantConfig(small=8, group_scale={"small": 32}), ("small", "2-D"))]
    for name, c, need in cases:
        f = os.path.join(d, "p5.rwkvq")
        try:
            api.quantize(CK, f, tokenizer=None, config=c, verbose=False)
            check("P5 %s -> отказ до работы" % name, False, "не упал")
        except TokenizerRequired:
            check("P5 %s -> отказ до работы" % name, False, "сначала дошло до токенизатора")
        except ValueError as e:
            check("P5 %s -> отказ до работы" % name, all(s in str(e) for s in need) and "не пишет" in str(e), str(e)[:200])
        except Exception as e:
            check("P5 %s -> отказ до работы" % name, False, "%s: %s" % (type(e).__name__, e))
        check("P5 %s: файл не создан" % name, not os.path.exists(f))

    # P6
    f = os.path.join(d, "p6.rwkvq")
    try:
        api.quantize(CK, f, tokenizer=None, real_gw=False, verbose=False,
                     config=QuantConfig(proj=7, group_scale={"proj": 32}, group_scale_mode={"proj": "asym_sb6"}))
        check("P6 real_gw=False пишет sb6 на 7 битах", os.path.getsize(f) > 1e6)
    except Exception as e:
        check("P6 real_gw=False пишет sb6 на 7 битах", False, "%s: %s" % (type(e).__name__, e))

    # P7
    def made(mode, **kw):
        try:
            QuantConfig(proj=6, group_scale={"proj": 32}, group_scale_mode={"proj": mode}, **kw); return True
        except ValueError:
            return False
    check("P7 режимы GW_MODES принимаются", all(made(m) for m in G.GW_MODES))
    check("P7 bogus и sym_typo -- ValueError", not made("bogus") and not made("sym_typo"), (made("bogus"), made("sym_typo")))
    check("P7 strict=False пропускает", made("bogus", strict=False))

    # P8
    c = QuantConfig(proj=6, group_scale={"proj": 32}, group_scale_mode={"proj": "sym_typo"}, strict=False)
    try:
        W.quantize_tensor(PROJ, SD[PROJ].float(), c, real_gw=True)
        check("P8 sym_typo писатель не принимает за sym", False, "записал")
    except NotImplementedError:
        check("P8 sym_typo писатель не принимает за sym", True)
    except Exception as e:
        check("P8 sym_typo писатель не принимает за sym", False, "%s: %s" % (type(e).__name__, e))
    got = W.unsupported_gw(CK, c)
    check("P8 unsupported_gw называет sym_typo", bool(got) and all(x[1] == "proj" and "sym_typo" in x[4] for x in got), got[:1])
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
        T = G.REAL_GW_SUPPORT
        def m_sb7(): T["sb6"] = (T["sb6"][0], (4, 5, 6, 7))
        def m_typo(): T["sym"] = (T["sym"][0] + ("sym_typo",), T["sym"][1]); G.GW_MODES["sym_typo"] = (False, "мутация")
        def m_bogus(): G.GW_MODES["bogus"] = (False, "мутация")
        muts = [("sb6 пишет 7 бит", m_sb7, "P2"), ("sym_typo внесён в таблицу", m_typo, "P8"),
                ("bogus внесён в GW_MODES", m_bogus, "P7"), ("суперблок sb6 8 -> 4", lambda: setattr(G, "SB6_SUPER", 4), "P1"),
                ("транспонирование выключено", lambda: setattr(W, "_transposes", lambda key, ndim: False), "P3"),
                ("отказ в api выключен", lambda: setattr(api, "_refuse_unsupported_gw", lambda *a, **k: None), "P5")]
        for name, apply, expect in muts:
            saved = (dict(T), dict(G.GW_MODES), G.SB6_SUPER, W._transposes, api._refuse_unsupported_gw)
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                T.clear(); T.update(saved[0]); G.GW_MODES.clear(); G.GW_MODES.update(saved[1])
                G.SB6_SUPER, W._transposes, api._refuse_unsupported_gw = saved[2:]
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
