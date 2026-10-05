"""06.10, находка 5 проверки API: потоковый префилл (128 разом + 128 по одному) против цельного (256 разом).
Вопрос: max|dlogit| 0.17-0.19 на последней позиции -- шум арифметики или ошибка. Прибор: KL по позициям 128..255 между плечами
и расстояние КАЖДОГО плеча до независимого эталона (torch fp32 на деквантованных весах файла; законы 37-38).
Плечи: нарезка тем же путём префилла (128+128, 128+64x2) -- пол перестановки; декод T=1 по умолчанию; декод с выключенными
LORA_Q / FUSE / FUSE_TAIL -- поиск источника. Колесо /tmp/rq_site_p1, cwd /tmp.
    cd /tmp && PYTHONPATH=/tmp/rq_site_p1 python .../p5_stream_0610.py <файл.rwkvq> <журнал.json>"""
import json, os, sys
import numpy as np, torch
import mlx.core as mx
import rwkv_quant
assert "/tmp/rq_site_p1" in rwkv_quant.__file__, rwkv_quant.__file__
from rwkv_quant.formats.reader import load_raw, load_dequantized
from rwkv_quant.backends.metal import quant_model as qm
from rwkv_quant.models.rwkv7_ref import RWKV7Ref
FILE, OUTJ = sys.argv[1], sys.argv[2]
W = os.path.expanduser("~/Develop/WKV-kvant/"); CK = sys.argv[3] if len(sys.argv) > 3 else W + "rwkv7-g1d-0.1b.pth"
NOTRUTH = len(sys.argv) > 4   # крупная модель: без fp32-эталона на весах файла (6 ГБ временного файла); колонка KL(эт||пл) тогда = KL(исх||пл)
ev = torch.load(W + "eval_text_heldout.pt", weights_only=False)["tokens"]
WINS = {"en": ev[0, :256], "ru": ev[12, :256], "sr": ev[24, :256]}
P0, T = 128, 256

def lsm(a): return torch.log_softmax(torch.from_numpy(np.asarray(a, dtype=np.float32)), -1)
def kl(lp, lq): return float((lp.exp() * (lp - lq)).sum(-1).mean())

# независимый эталон: torch fp32 на деквантованных весах ЭТОГО файла; и исходный bf16-чекпоинт (масштаб квантования)
if not NOTRUTH:
    SD = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)
    dq = load_dequantized(FILE); out = {}
    for k, v in SD.items():
        w = dq[k]
        if tuple(w.shape) != tuple(v.shape): w = w.T if (w.dim() == 2 and tuple(w.T.shape) == tuple(v.shape)) else w.reshape(v.shape)
        out[k] = w.float().contiguous()
    tmp = "/tmp/rq_p1/deq5_%d.pth" % os.getpid(); os.makedirs("/tmp/rq_p1", exist_ok=True); torch.save(out, tmp); del out, dq
    MF = RWKV7Ref(tmp, device="cpu", dtype=torch.float32, compute_dtype=torch.float32); os.remove(tmp)
MO = RWKV7Ref(CK, device="cpu", dtype=torch.bfloat16, compute_dtype=torch.float32)
TRUTH, ORIG = {}, {}
with torch.no_grad():
    for n, wt in WINS.items():
        ORIG[n] = torch.log_softmax(MO.forward(wt[None], cfg=None).float(), -1)[0, P0:]
        TRUTH[n] = ORIG[n] if NOTRUTH else torch.log_softmax(MF.forward(wt[None], cfg=None).float(), -1)[0, P0:]
del MO

def build(**flags):
    old = {k: getattr(qm, k) for k in flags}
    for k, v in flags.items(): setattr(qm, k, v)
    m = qm.QuantRWKV7(load_raw(FILE))
    return m, old
def restore(old):
    for k, v in old.items(): setattr(qm, k, v)

def run(m, ids, splits, raw=False):
    """Логиты [T, V] fp32, нарезка splits; raw -- некомпилированный forward_stateful."""
    f = m.forward_stateful if raw else m.step
    st = m.init_state(1); outs = []; p = 0
    for s in splits:
        lg, st = f(ids[:, p:p + s], st); lg = lg.astype(mx.float32); mx.eval(lg, st)
        outs.append(np.array(lg)[0]); p += s
    return np.concatenate(outs, 0)

ARMS = [  # имя, флаги, нарезка, raw
    ("префилл 128+128 (тот же путь, другая нарезка)", {}, [128, 128], False),
    ("префилл 128+64x2", {}, [128] + [2] * 64, False),
    ("префилл 128+32x4", {}, [128] + [4] * 32, False),
    ("ДЕКОД 128+128x1, умолчания", {}, [128] + [1] * 128, False),
    ("декод, некомпилированный forward_stateful", {}, [128] + [1] * 128, True),
    ("декод, LORA_Q=None", dict(LORA_Q=None), [128] + [1] * 128, False),
    ("декод, FUSE_TAIL=False", dict(FUSE_TAIL=False), [128] + [1] * 128, False),
    ("декод, FUSE=False", dict(FUSE=False), [128] + [1] * 128, False),
    ("декод, LORA_Q=None + FUSE=False", dict(LORA_Q=None, FUSE=False), [128] + [1] * 128, False),
    ("декод по одному с нуля (256x1)", {}, [1] * 256, False),
]
RES = dict(file=FILE, arms=[])
m0, _ = build()
print("файл %s; пресет %s, fast_ln %s (%s); LORA_Q=%r FUSE=%r FUSE_TAIL=%r DECODE_VIEWS=%r" % (
    os.path.basename(FILE), getattr(m0, "preset", None), m0.fast_ln, m0.fast_ln_source, qm.LORA_Q, qm.FUSE, qm.FUSE_TAIL, qm.DECODE_VIEWS))
FULL, FULLL = {}, {}
for n, wt in WINS.items():
    ids = mx.array(wt[None].numpy().astype(np.int32))
    a = run(m0, ids, [T]); b = run(m0, ids, [T])
    assert np.array_equal(a, b), "цельный префилл не воспроизводится бит-в-бит"
    FULL[n] = a[P0:]; FULLL[n] = lsm(a[P0:])
sc = max(float(np.abs(FULL[n]).max()) for n in WINS)
print("логиты цельного: dtype шага fp16->fp32, max|logit| %.2f, ulp fp16 на этом масштабе %.4f" % (sc, float(np.spacing(np.float16(sc)))))
print("эталоны (позиции 128..255, среднее по en/ru/sr): KL(исходный bf16 || fp32-файл) = %.5f   [цена квантования]" % np.mean([kl(ORIG[n], TRUTH[n]) for n in WINS]))
print("                                                 KL(fp32-файл || цельный префилл Metal) = %.3e" % np.mean([kl(TRUTH[n], FULLL[n]) for n in WINS]))
RES["kl_quant"] = float(np.mean([kl(ORIG[n], TRUTH[n]) for n in WINS])); RES["kl_truth_full"] = float(np.mean([kl(TRUTH[n], FULLL[n]) for n in WINS]))
print("\n%-48s %9s %9s %11s %11s %11s %7s  рост max|d| по шагам 1/8/32/128" % ("плечо", "max|d|посл", "max|d|все", "KL(цел||пл)", "KL(эт||пл)", "KL(исх||пл)", "top1"))
for name, flags, splits, raw in ARMS:
    m, old = (m0, {}) if not flags else build(**flags)
    r = dict(name=name, per={})
    try:
        for n, wt in WINS.items():
            ids = mx.array(wt[None].numpy().astype(np.int32))
            a = run(m, ids, splits, raw)[P0:]; la = lsm(a); d = np.abs(a - FULL[n])
            r["per"][n] = dict(d_last=float(d[-1].max()), d_all=float(d.max()), kl_full=kl(FULLL[n], la), kl_truth=kl(TRUTH[n], la),
                               kl_orig=kl(ORIG[n], la), top1=float((la.argmax(-1) == FULLL[n].argmax(-1)).float().mean()),
                               growth=[float(d[i].max()) for i in (0, 7, 31, 127)])
    finally:
        restore(old)
    ag = lambda k: float(np.mean([r["per"][n][k] for n in WINS])); mxg = lambda k: float(np.max([r["per"][n][k] for n in WINS]))
    r.update(d_last=mxg("d_last"), d_all=mxg("d_all"), kl_full=ag("kl_full"), kl_truth=ag("kl_truth"), kl_orig=ag("kl_orig"), top1=ag("top1"))
    g = np.max([r["per"][n]["growth"] for n in WINS], 0)
    print("%-48s %9.4f %9.4f %11.3e %11.3e %11.5f %7.4f  %.4f / %.4f / %.4f / %.4f" % (name, r["d_last"], r["d_all"], r["kl_full"], r["kl_truth"], r["kl_orig"], r["top1"], *g), flush=True)
    RES["arms"].append(r); json.dump(RES, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    if flags: del m
print("опора: KL(исх||цельный) = %.5f" % np.mean([kl(ORIG[n], FULLL[n]) for n in WINS]))
print("ГОТОВО")
