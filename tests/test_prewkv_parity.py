"""Гейт ядра пред-WKV: prewkv_kernel против prewkv_ref.

ПОРОГИ ЗАРЕГИСТРИРОВАНЫ 10.09, ДО ПЕРВОГО ЗАМЕРА, и задним числом не
двигаются:

    выходы k и v                БИТ-В-БИТ, без допуска
    w, kk, kk*a                 relmax <= 1e-6
    дрейф состояния WKV, 1024   <= 1e-3 относительных И рост не быстрее
                                линейного
    KL(фьюз || нефьюз)          <= 1e-4 нат/ток и >= 100x меньше KL
                                квантования
    greedy 256 токенов          совпадение
    ppl                         СПРАВОЧНО, решать по ней нельзя

ПРО `a`: отдельным выходом её нет, она входит множителем в kk*a и
слагаемым в k. Порог на kk*a ограничивает её сверху, а требование
бит-в-бит на k -- строже любого допуска на неё.

ПРО ОПАСНОСТЬ ИМЕННО ЭТОГО ЯДРА: w -- коэффициент затухания, он
умножает состояние на КАЖДОМ шаге, и относительная ошибка компаундится
как (1+eps)^n по длине контекста. Проверка на одном шаге её не видит --
отсюда раздел про дрейф на 1024 токенах, teacher-forced (иначе
расхождение траектории токенов подменит собой расхождение ядра).

ЭТАЛОН ДЛЯ ЯДРА -- ТОТ ЖЕ КОД, КОТОРЫМ СЧИТАЕТ МОДЕЛЬ. Гейт зовёт
fused_prewkv.prewkv_ref, а не свою копию формул: копия разошлась бы с
моделью молча (закон 30). Пока prewkv_kernel -- псевдоним эталона, гейт
обязан проходить ТРИВИАЛЬНО; если он краснеет в этом состоянии, красен
сам гейт.

ОТКЛОНЕНИЕ ОТ ПЛАНА 10.09, СОЗНАТЕЛЬНОЕ: плечи сравниваются ПООКОННО
в одном процессе, без сброса логитов на диск. Пара логитов одного окна
-- 268 МБ, весь корпус -- 5 ГБ; пооконное чередование даёт ту же парную
разность при постоянной памяти и без файла, который надо чистить между
размерами. Скорость гейт не меряет, поэтому закон 1 тут ни при чём.

Запуск: по одному размеру на процесс, из-под screen (закон 13 и таймаут
MCP). Ручки: RWKVQ_GATE_NSEQ (окон KL, по умолчанию 38 -- закон 9),
RWKVQ_GATE_DRIFT (1024), RWKVQ_GATE_GREEDY (256), RWKVQ_GATE_SKIP
(список разделов через запятую: comp,drift,greedy,kl).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm
from rwkv_quant.backends.metal import fused_prewkv as fp

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Develop/WKV-kvant/compression_v2_cand.rwkvq")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
NSEQ = int(os.environ.get("RWKVQ_GATE_NSEQ", "38"))
NDRIFT = int(os.environ.get("RWKVQ_GATE_DRIFT", "1024"))
NGREEDY = int(os.environ.get("RWKVQ_GATE_GREEDY", "256"))
SKIP = set(x for x in os.environ.get("RWKVQ_GATE_SKIP", "").split(",") if x)

FAILS = []
T0 = time.time()


def say(msg):
    print("[%6.1f] %s" % (time.time() - T0, msg), flush=True)


def check(name, ok, detail):
    say(("OK   " if ok else "FAIL ") + name + ": " + detail)
    if not ok:
        FAILS.append(name)


def f32(a):
    return np.array(a.astype(mx.float32))


def relmax(a, b):
    x, y = f32(a), f32(b)
    d = float(np.max(np.abs(x - y)))
    return d / (float(np.max(np.abs(x))) + 1e-9)


def bitsame(a, b):
    if a.dtype != b.dtype:
        return False
    return np.array_equal(f32(a), f32(b))


say("модель: %s" % MODEL)
model = qm.QuantRWKV7(load_raw(MODEL))
blob = torch.load(CORPUS)
TOK = blob["tokens"].numpy().astype(np.int32)
LANG = blob["lang"]
say("корпус: %d окон по %d, слоёв в модели %d" % (
    TOK.shape[0], TOK.shape[1], model.n_layer))
say("флаги: FUSE=%s FUSE_TAIL=%s LORA_Q=%r EMB_GATHER=%s" % (
    qm.FUSE, qm.FUSE_TAIL, qm.LORA_Q, qm.EMB_GATHER))
say("kernel is ref: %s" % (fp.prewkv_kernel is fp.prewkv_ref))


# --- мутация: РАЗРЕШЕНИЕ ГЕЙТА, а не проверка ядра (закон 37) --------
# Гейт, проходящий тривиально, обязан быть померен мутацией: иначе
# неизвестно, какую НАИМЕНЬШУЮ ошибку он вообще ловит, и зелёный цвет
# ничего не значит. RWKVQ_GATE_MUTATE=eps умножает один выход ядра на
# (1+eps); RWKVQ_GATE_MUTATE_WHAT выбирает какой (w, k, v, nkk, kka).
# Измеренные границы записаны в шапке результатов сессии.
# ГРАНИЦЫ ПРИМЕНИМОСТИ, ПОМЕРЕНЫ МУТАЦИЕЙ (0.1B, 12 слоёв, дрейф 128):
#   k, v (бит-в-бит)  -- ловят 1e-7 и ниже: разрешение упирается только
#                        в то, меняет ли мутация хоть один бит fp16;
#   w  relmax         -- ловит 1e-5, зелен на 1e-6; но зелен потому,
#                        что мутация 1e-6 в fp16 НИЧЕГО НЕ МЕНЯЕТ, а не
#                        потому, что гейт слеп;
#   nkk, kka relmax   -- ловят 1e-6, зелены на 1e-7;
#   дрейф <= 1e-3     -- ловит 1e-3 по w и 1e-2 по nkk/kka, то есть на
#                        два порядка грубее компонентного паритета;
#   greedy 256        -- ловит только 1e-2, самый слабый прибор набора.
# ВАЖНОЕ СЛЕДСТВИЕ ПРО ПОРОГ 1e-6 НА fp16-ВЫХОДАХ: решётка fp16 даёт
# относительный шаг около 4.9e-4 на элементе масштаба единицы, то есть
# ОДИН ulp на крупном элементе w уже даёт relmax сильно выше 1e-6.
# Порог 1e-6 на этих выходах фактически требует бит-в-бит и допускает
# разве что ulp на элементе, мелком относительно максимума тензора
# (закон 37: порог под решёткой типа проверяет бит-в-бит).
# ПОЛ ПРИБОРА ДРЕЙФА: при тождественных плечах дрейф РОВНО 0; мутация
# в один ulp сразу даёт около 1e-4 и дальше НЕ РАСТЁТ, а выходит на
# плато. Ниже этого пола отношения соседних отметок -- шум.
DRIFT_FLOOR = 3e-4
MUT = float(os.environ.get("RWKVQ_GATE_MUTATE", "0") or 0)
MUT_WHAT = os.environ.get("RWKVQ_GATE_MUTATE_WHAT", "w")
if MUT:
    _MIDX = {"w": 0, "k": 1, "v": 2, "nkk": 3, "kka": 4}[MUT_WHAT]

    def _mutated(*a):
        out = list(fp.prewkv_ref(*a))
        t = out[_MIDX]
        out[_MIDX] = (t.astype(mx.float32) * (1.0 + MUT)).astype(t.dtype)
        return tuple(out)

    fp.prewkv_kernel = _mutated
    qm.prewkv_kernel = _mutated
    say("МУТАЦИЯ %s x (1+%g): гейт ОБЯЗАН покраснеть" % (MUT_WHAT, MUT))


# --- 1. компонентный паритет на РЕАЛЬНЫХ входах, все слои ---------------
if "comp" not in SKIP:
    CAP = []
    _orig = qm.QuantTMix._prewkv

    def _cap(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype):
        CAP.append((y_w, y_a, y_v, k, v, v_first, self.w_lora_B_b,
                    self.a_lora_B_b, self.v_lora_B_b, self.k_k, self.k_a,
                    self.layer_id, B, T, self.H, self.S, dtype))
        return _orig(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype)

    qm.QuantTMix._prewkv = _cap
    prompt = mx.array(TOK[0:1, :64])
    st = model.init_state(1)
    logits, st = model.forward_stateful(prompt, st, last_only=True)
    mx.eval(logits)
    CAP.clear()          # префилл выбрасываем: ядро работает только при T=1
    tok = mx.argmax(logits[:, -1], axis=-1)
    logits, st = model.forward_stateful(tok[None], st)
    mx.eval(logits)
    qm.QuantTMix._prewkv = _orig
    say("захвачено входов пред-WKV на decode-шаге: %d" % len(CAP))
    check("захват покрыл все слои", len(CAP) == model.n_layer,
          "%d из %d" % (len(CAP), model.n_layer))

    NAMES = ("w", "k", "v", "nkk", "kka", "vfirst")
    TOL = {"w": 1e-6, "nkk": 1e-6, "kka": 1e-6}
    worst = {k: 0.0 for k in TOL}
    worst_at = {k: -1 for k in TOL}
    bad = {"k": [], "v": [], "vfirst": []}
    degen = 0
    for args in CAP:
        lid = args[11]
        ra = fp.prewkv_ref(*args)
        rb = fp.prewkv_kernel(*args)
        mx.eval(*[t for t in tuple(ra) + tuple(rb) if t is not None])
        if float(np.max(np.abs(f32(ra[0])))) == 0.0:
            degen += 1
        for nm, a, b in zip(NAMES, ra, rb):
            if a is None or b is None:
                continue
            if nm in TOL:
                r = relmax(a, b)
                if r > worst[nm]:
                    worst[nm], worst_at[nm] = r, lid
            elif not bitsame(a, b):
                bad[nm].append(lid)
    check("вход не вырожден (w != 0)", degen == 0,
          "вырожденных слоёв %d" % degen)
    for nm in ("k", "v", "vfirst"):
        check("%s бит-в-бит" % nm, not bad[nm],
              "расхождений 0" if not bad[nm] else "слои %r" % bad[nm][:8])
    for nm in ("w", "nkk", "kka"):
        check("%s relmax <= 1e-6" % nm, worst[nm] <= TOL[nm],
              "%.3e (худший слой %d)" % (worst[nm], worst_at[nm]))


# --- 2. дрейф состояния WKV, teacher-forced ------------------------------
if "drift" not in SKIP:
    seq = TOK.reshape(-1)[:NDRIFT]
    marks = []
    m = 32
    while m <= NDRIFT:
        marks.append(m)
        m *= 2
    if marks[-1] != NDRIFT:
        marks.append(NDRIFT)

    def run_forced(fuse):
        qm.FUSE_PREWKV = fuse
        st = model.init_state(1)
        snap = {}
        for i in range(len(seq)):
            _, st = model.forward_stateful(
                mx.array(seq[i:i + 1].reshape(1, 1)), st)
            if (i + 1) in marks:
                mx.eval(*[s[0] for s in st])
                snap[i + 1] = [f32(s[0]) for s in st]
        return snap

    say("дрейф: %d шагов на плечо, отметки %r" % (NDRIFT, marks))
    sa = run_forced(False)
    say("плечо без ядра пройдено")
    sb = run_forced(True)
    qm.FUSE_PREWKV = False
    say("плечо с ядром пройдено")

    drift = {}
    for n in marks:
        r = 0.0
        for a, b in zip(sa[n], sb[n]):
            d = float(np.max(np.abs(a - b)))
            r = max(r, d / (float(np.max(np.abs(a))) + 1e-9))
        drift[n] = r
    say("дрейф по отметкам: " + ", ".join(
        "%d:%.3e" % (n, drift[n]) for n in marks))
    check("дрейф на %d <= 1e-3" % NDRIFT, drift[NDRIFT] <= 1e-3,
          "%.3e" % drift[NDRIFT])
    ratios = []
    for i in range(1, len(marks)):
        prev, cur = drift[marks[i - 1]], drift[marks[i]]
        ratios.append(cur / prev if prev > 0 else (0.0 if cur == 0 else 1e9))
    step = marks[1] / marks[0] if len(marks) > 1 else 2.0
    lim = 1.25 * step
    say("отношения соседних отметок: " + ", ".join("%.2f" % x for x in ratios))
    if drift[NDRIFT] <= DRIFT_FLOOR:
        check("рост не быстрее линейного", True,
              "дрейф %.3e ниже пола инструмента %.0e -- рост не "
              "оценивается, отношения там шум" % (drift[NDRIFT], DRIFT_FLOOR))
    else:
        check("рост не быстрее линейного", all(x <= lim for x in ratios),
              "max %.2f при линейном %.2f (потолок %.2f)" % (
                  max(ratios) if ratios else 0.0, step, lim))


# --- 3. greedy ----------------------------------------------------------
if "greedy" not in SKIP:
    def greedy(fuse, n):
        qm.FUSE_PREWKV = fuse
        st = model.init_state(1)
        logits, st = model.forward_stateful(mx.array(TOK[0:1, :64]), st,
                                            last_only=True)
        out = []
        for _ in range(n):
            t = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(t)
            out.append(int(t[0]))
            logits, st = model.forward_stateful(t.reshape(1, 1), st)
        return out

    ga = greedy(False, NGREEDY)
    gb = greedy(True, NGREEDY)
    qm.FUSE_PREWKV = False
    same = sum(1 for x, y in zip(ga, gb) if x == y)
    first = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), -1)
    check("greedy %d совпадает" % NGREEDY, same == NGREEDY,
          "%d/%d, первое расхождение на %d" % (same, NGREEDY, first))


# --- 4. KL между плечами, ПОТОКЕННО (ядро живёт только на декоде) -------
# Прибор -- ablate_subgroups.kl_stats, ТОТ ЖЕ, которым посчитаны все KL
# проекта: fp64, log_softmax чанками по 64 позиции. Своей формулы здесь
# нет намеренно: отношение к KL квантования иначе считалось бы двумя
# разными линейками (закон 39).
#
# ЛОВУШКА 09.09: обвязка KL по умолчанию берёт срез [:8], а корпус
# отсортирован по языкам -- первые 20 окон русские. Здесь окна берутся
# подряд от нуля до NSEQ, и умолчание NSEQ = 38, то есть весь корпус
# (закон 9). Уменьшать NSEQ можно только понимая, что мерится русский.
UNCHECKED = []
if "kl" not in SKIP:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ablate_subgroups import kl_stats, boot_ci

    QREF = os.environ.get("RWKVQ_GATE_QKL", "")
    ref = None
    if QREF and os.path.exists(QREF):
        ref = np.load(QREF, mmap_mode="r")
        say("эталон квантования: %s %r" % (os.path.basename(QREF), ref.shape))
    elif QREF:
        say("ВНИМАНИЕ: RWKVQ_GATE_QKL=%s не существует" % QREF)
    nw = min(NSEQ, TOK.shape[0])
    if ref is not None:
        nw = min(nw, ref.shape[0])
    say("KL: %d окон по %d, плечи чередуются потокенно" % (nw, TOK.shape[1]))

    def nll_of(M, tgt):
        acc = 0.0
        for a in range(0, M.shape[0], 64):
            b = min(a + 64, M.shape[0])
            Pm = M[a:b].astype(np.float64)
            mx_ = Pm.max(1, keepdims=True)
            lse = np.log(np.exp(Pm - mx_).sum(1, keepdims=True)) + mx_
            hit = Pm[np.arange(b - a), tgt[a:b]]
            acc += float((lse[:, 0] - hit).sum())
        return acc

    per_arm, per_q = [], []
    nll_a = nll_b = 0.0
    ntok = 0
    for i in range(nw):
        row = TOK[i]
        L = len(row) - 1
        stA = model.init_state(1)
        stB = model.init_state(1)
        A_ = B_ = None
        for t in range(L):
            x = mx.array(row[t:t + 1].reshape(1, 1))
            qm.FUSE_PREWKV = False
            la, stA = model.forward_stateful(x, stA)
            qm.FUSE_PREWKV = True
            lb, stB = model.forward_stateful(x, stB)
            mx.eval(la, lb)
            va, vb = f32(la[0, -1]), f32(lb[0, -1])
            if A_ is None:
                A_ = np.empty((L, va.shape[0]), dtype=np.float32)
                B_ = np.empty((L, va.shape[0]), dtype=np.float32)
            A_[t], B_[t] = va, vb
        kl_arm, _ = kl_stats(A_, B_)
        per_arm.append(float(kl_arm.mean()))
        if ref is not None:
            kq, _ = kl_stats(np.asarray(ref[i]), A_)
            per_q.append(float(kq.mean()))
        tgt = row[1:1 + L].astype(np.int64)
        nll_a += nll_of(A_, tgt)
        nll_b += nll_of(B_, tgt)
        ntok += L
        say("  окно %2d (%s): KL плеч %.3e%s" % (
            i, LANG[i], per_arm[-1],
            "" if ref is None else ", KL квантования %.6f" % per_q[-1]))
        A_ = B_ = None
    qm.FUSE_PREWKV = False

    kl_mean = float(np.mean(per_arm))
    lo, hi = boot_ci(per_arm)
    check("KL(фьюз || нефьюз) <= 1e-4 нат/ток", kl_mean <= 1e-4,
          "%.6e [%.2e; %.2e], худшее окно %.3e, %d токенов" % (
              kl_mean, lo, hi, max(per_arm), ntok))
    if ref is not None:
        qkl = float(np.mean(per_q))
        ratio = float("inf") if kl_mean <= 0 else qkl / kl_mean
        check("запас против KL квантования >= 100x", ratio >= 100.0,
              "KL квантования %.6f, отношение %.1fx" % (qkl, ratio))
    else:
        UNCHECKED.append("запас 100x против KL квантования: "
                         "RWKVQ_GATE_QKL не задан")
    say("ppl СПРАВОЧНО (решать по ней нельзя): нефьюз %.4f, фьюз %.4f" % (
        float(np.exp(nll_a / ntok)), float(np.exp(nll_b / ntok))))


# --- итог ---------------------------------------------------------------
say("=" * 68)
for u in UNCHECKED:
    say("НЕ ПРОВЕРЕНО: " + u)
if FAILS:
    say("ГЕЙТ КРАСНЫЙ: " + ", ".join(FAILS))
    sys.exit(1)
say("ГЕЙТ ЗЕЛЁНЫЙ" + (" (с непроверенными пунктами)" if UNCHECKED else ""))
sys.exit(2 if UNCHECKED else 0)
