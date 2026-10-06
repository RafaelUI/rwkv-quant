"""06.10: проверка публичного API «неверным использованием» -- на установленном колесе, cwd=/tmp (как api_smoke_0410).
Каждый случай: что вернулось -- исключение (тип + первая строка) или «принято» (+ создан ли файл, были ли предупреждения,
что напечатано). Оценка «ладно / плохо» -- глазами в отчёте; скрипт ничего не утверждает, кроме того, что сам не падает.
    cd /tmp && PYTHONPATH=<site> python .../api_misuse_0610.py <журнал.json> [cuda]"""
import contextlib, copy, io, json, os, sys, time, warnings
import torch
import rwkv_quant
from rwkv_quant import quantize, QuantConfig, presets
assert "/Develop/rwkv-quant" not in rwkv_quant.__file__, rwkv_quant.__file__
OUTJ = sys.argv[1]; CUDA = len(sys.argv) > 2
if CUDA:
    CK = os.path.expanduser("~/rwkvq/ckpt/rwkv7-g1d-0.1b-20260129-ctx8192.pth"); TOK = os.path.expanduser("~/rwkvq/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"); DEV = "cuda:0"
else:
    W = os.path.expanduser("~/Develop/WKV-kvant/"); CK = W + "rwkv7-g1d-0.1b.pth"; TOK = "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt"; DEV = None
O = "/tmp/rq_misuse_%d/" % int(time.time()); os.makedirs(O)
from rwkv_quant.calibration import act_stats as A
A.CACHE_DIR = O + "cache_act"                                  # рабочий кеш не трогаем и не засоряем
from rwkv_quant.calibration import autopick as ap
ap.MEASURE_CACHE = O + "cache_measure"
RES = []; N = [0]
FAST = dict(gptq=False, autopick=False, verbose=False)

def case(name, fn, out=None):
    N[0] += 1; out = out or (O + "c%02d.rwkvq" % N[0]); rec = dict(n=N[0], name=name); t0 = time.time(); buf = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(buf):
            warnings.simplefilter("always"); r = fn(out)
        rec.update(result="ПРИНЯТО", file=os.path.exists(out), mb=round(os.path.getsize(out) / 1e6, 1) if os.path.exists(out) else None,
                   warnings=[str(w.message)[:200] for w in ws], ret=(repr(r)[:120] if r is not None else None))
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt): raise
        rec.update(result=type(e).__name__, msg=str(e).strip().split("\n")[0][:300], file=os.path.exists(out))
    rec["stdout"] = [l for l in buf.getvalue().split("\n") if l.strip() and "RWKV7Ref" not in l][:3]; rec["sec"] = round(time.time() - t0, 1)
    RES.append(rec); json.dump(RES, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    tail = (rec.get("msg") or "файл %s%s%s" % ("%.1f МБ" % rec["mb"] if rec.get("file") else "НЕ создан", "; предупреждения: %s" % rec["warnings"] if rec.get("warnings") else "", "; печать: %s" % rec["stdout"][:1] if rec["stdout"] else ""))
    print("%2d. %-58s -> %-18s %s%s" % (N[0], name, rec["result"], tail[:230], "  [ФАЙЛ ОСТАЛСЯ]" if rec["result"] != "ПРИНЯТО" and rec.get("file") else ""), flush=True)

q = lambda out, **kw: quantize(kw.pop("ck", CK), out, **dict(dict(tokenizer=TOK, device=DEV, **FAST), **kw))
C = lambda **kw: copy.deepcopy(presets.COMPRESSION)

print("== пути и файлы")
case("чекпоинта нет", lambda o: q(o, ck="/tmp/нет_такого.pth"))
case("чекпоинт -- не RWKV-7 (случайный .pth)", lambda o: (torch.save({"a.weight": torch.randn(8, 8)}, O + "junk.pth"), q(o, ck=O + "junk.pth"))[1])
case("чекпоинт -- текстовый файл", lambda o: q(o, ck=TOK))
case("каталога для выхода нет", lambda o: q(o), out=O + "нет_каталога/x.rwkvq")
case("выход == входной чекпоинт", lambda o: (torch.save(torch.load(CK, map_location="cpu", mmap=True, weights_only=True), O + "copy.pth"), q(O + "copy.pth", ck=O + "copy.pth"))[1], out=O + "copy.pth")
print("== токенизатор")
case("tokenizer=None с пресетом", lambda o: q(o, tokenizer=None))
case("tokenizer -- путь, которого нет", lambda o: q(o, tokenizer="/tmp/нет_словаря.txt"))
case("tokenizer -- путь к НЕ словарю (корпус)", lambda o: q(o, tokenizer=A.CORPUS))
case("tokenizer -- число", lambda o: q(o, tokenizer=42))
case("tokenizer -- функция, возвращает строки", lambda o: q(o, tokenizer=lambda s: s.split()))
case("tokenizer -- функция, id за пределами vocab", lambda o: q(o, tokenizer=lambda s: [70000 + (b % 7) for b in s.encode()]))
case("tokenizer -- функция, пустой список", lambda o: q(o, tokenizer=lambda s: []))
case("tokenizer -- чужой словарь в пределах vocab (байты)", lambda o: q(o, tokenizer=lambda s: list(s.encode())))
print("== пресет и конфиг")
case("preset с опечаткой", lambda o: q(o, preset="compresion"))
case("preset=None", lambda o: q(o, preset=None))
case("config -- словарь вместо QuantConfig", lambda o: q(o, config={"proj": 4}))
case("config -- строка 'compression'", lambda o: q(o, config="compression"))
case("config -- только биты (пример старого docstring)", lambda o: q(o, config=QuantConfig(proj=4, cmix=4, emb=6, head=6)))
case("QuantConfig: опечатка в группе", lambda o: q(o, config=QuantConfig(prooj=4)))
case("QuantConfig: биты 3", lambda o: q(o, config=QuantConfig(proj=3, group_scale={"proj": 32}, group_scale_mode={"proj": "asym_sb6"})))
case("QuantConfig: биты 0", lambda o: q(o, config=QuantConfig(proj=0)))
case("QuantConfig: биты 12 построчно", lambda o: q(o, config=QuantConfig(proj=12)))
case("QuantConfig: биты строкой '4'", lambda o: q(o, config=QuantConfig(proj="4", group_scale={"proj": 32}, group_scale_mode={"proj": "asym_sb6"})))
case("QuantConfig: биты 4.0 (float)", lambda o: q(o, config=QuantConfig(proj=4.0, group_scale={"proj": 32}, group_scale_mode={"proj": "asym_sb6"})))
case("QuantConfig: неизвестный режим", lambda o: q(o, config=QuantConfig(proj=4, group_scale={"proj": 32}, group_scale_mode={"proj": "sb7"})))
case("QuantConfig: group_scale 33 (не делит ширину)", lambda o: q(o, config=QuantConfig(proj=4, group_scale={"proj": 33}, group_scale_mode={"proj": "asym_sb6"})))
case("QuantConfig: group_scale -32", lambda o: q(o, config=QuantConfig(proj=4, group_scale={"proj": -32}, group_scale_mode={"proj": "asym_sb6"})))
case("QuantConfig: outlier_fracs 1.5", lambda o: q(o, config=QuantConfig(proj=8, outlier_fracs={"proj": 1.5})))
case("QuantConfig: clip_percentiles 250", lambda o: q(o, config=QuantConfig(proj=8, clip_percentiles={"proj": 250})))
def _ov(o):
    c = C(); c.bits_overrides = {"нет_такой_матрицы": 8}; return q(o, config=c)
case("bits_overrides: шаблон ни с чем не совпал", _ov)
def _ov2(o):
    c = C(); c.bits_overrides = {"blocks.0.att.key.weight": 7}; return q(o, config=c)
case("bits_overrides: 7 бит в режиме sb6", _ov2)
print("== прочие аргументы quantize()")
case("act_stats -- путь, которого нет", lambda o: q(o, act_stats="/tmp/нет.pt"))
case("act_stats -- файл не того формата", lambda o: (torch.save({"x": 1}, O + "bad_act.pt"), q(o, act_stats=O + "bad_act.pt"))[1])
case("act_stats=True", lambda o: q(o, act_stats=True))
case("device='gpu'", lambda o: q(o, device="gpu", gptq=True))
case("device='cuda:7'", lambda o: q(o, device="cuda:7", gptq=True))
case("device=0 (число)", lambda o: q(o, device=0, gptq=True))
case("gptq=True со своим конфигом без sb6", lambda o: q(o, config=QuantConfig(proj=8), gptq=True))
case("gptq=True при real_gw=False", lambda o: q(o, preset="compression", gptq=True, real_gw=False))
case("gptq_calib -- 8 окон", lambda o: q(o, gptq=True, gptq_calib=torch.zeros(8, 512, dtype=torch.long)))
case("gptq_calib -- путь, которого нет", lambda o: q(o, gptq=True, gptq_calib="/tmp/нет.pt"))
case("gptq_calib -- список строк", lambda o: q(o, gptq=True, gptq_calib=["текст"] * 600))
case("gptq_calib -- токены за пределами vocab", lambda o: q(o, gptq=True, gptq_calib=torch.full((600, 64), 70000)))
case("autopick=True со своим конфигом без AW", lambda o: q(o, config=QuantConfig(proj=8), autopick=True))
case("autopick_budget=-1", lambda o: q(o, preset="compression", autopick=True, autopick_budget=-1, measure=O + "нет.json"))
case("measure -- путь, которого нет", lambda o: q(o, preset="compression", autopick=True, measure="/tmp/нет.json"))
case("measure -- JSON не того вида", lambda o: (open(O + "bad.json", "w").write("{}"), q(o, preset="compression", autopick=True, measure=O + "bad.json"))[1])
case("measure='yes'", lambda o: q(o, preset="compression", autopick=True, measure="yes"))
case("autopick='да' (строка)", lambda o: q(o, preset="reduction", autopick="да"))
case("неизвестный именованный аргумент", lambda o: q(o, bits=4))
case("allow_per_row=True с конфигом «только биты»", lambda o: q(o, config=QuantConfig(proj=4, cmix=4), allow_per_row=True))
if not CUDA:
    print("== чтение и рантайм (Metal)")
    import mlx.core as mx
    from rwkv_quant.formats.reader import load_raw
    import rwkv_quant.backends.metal.quant_model as qm
    from rwkv_quant.backends.metal import generate as G
    good = O + "good.rwkvq"; q(good, preset="reduction")
    fake = O + "fake.rwkvq"; q(fake, preset="reduction", real_gw=False)
    case("load_raw: файла нет", lambda o: load_raw("/tmp/нет.rwkvq"))
    case("load_raw: это .pth, а не .rwkvq", lambda o: load_raw(CK))
    case("QuantRWKV7 от пути (строка вместо load_raw)", lambda o: qm.QuantRWKV7(good))
    case("QuantRWKV7 от fake-файла (real_gw=False)", lambda o: qm.QuantRWKV7(load_raw(fake)))
    m = qm.QuantRWKV7(load_raw(good))
    case("generate: пустой промпт", lambda o: G.generate(m, [], 4))
    case("generate: id за пределами vocab", lambda o: G.generate(m, [70000, 5], 4))
    case("generate: отрицательный id", lambda o: G.generate(m, [-1, 5], 4))
    case("generate: промпт строкой", lambda o: G.generate(m, "привет", 4))
    case("generate: n=0", lambda o: G.generate(m, [5, 6], 0))
    case("generate: n=-3", lambda o: G.generate(m, [5, 6], -3))
    case("step: одномерные ids", lambda o: m.step(mx.array([5, 6, 7]), m.init_state(1)))
    case("step: состояние от другого батча", lambda o: m.step(mx.array([[5, 6]]), m.init_state(2)))
    case("step: state=None", lambda o: m.step(mx.array([[5, 6]]), None))
    case("QuantRWKV7(lora_q='да')", lambda o: qm.QuantRWKV7(load_raw(good), lora_q="да").lora_q)
print("ГОТОВО, случаев %d, каталог %s" % (N[0], O))
