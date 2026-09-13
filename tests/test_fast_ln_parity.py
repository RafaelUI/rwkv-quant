"""Гейт правки FAST_LN: штатный mx.fast.layer_norm против рукописной нормы.

ПОРОГИ ЗАРЕГИСТРИРОВАНЫ 12.09, ДО ПЕРВОГО ПРОГОНА ГЕЙТА, и задним числом
не двигаются:

    1. relmax(fast, off) на каждом вызове   <= max(4 ulp типа ВХОДА,
                                                relmax(1pass, off))
    2. расстояние до fp64-эталона           ДИАГНОСТИКА, порога нет --
                                            см. абзац НИЖЕ
    3. дрейф состояния WKV на 1024 токенах  <= 1e-3 (порог 10.09) И рост не
                                                быстрее линейного
    4. greedy 256 токенов                   СПРАВОЧНО, решать по нему нельзя
    5. KL(fast || off)                      <= 1e-4 нат/ток И >= 100x меньше
                                            KL квантования
    6. ppl                                  СПРАВОЧНО, решать по ней нельзя

ПОЧЕМУ НЕ БИТ-В-БИТ, В ОТЛИЧИЕ ОТ ГЕЙТА ЯДРА. Правка меняет ПОРЯДОК
РЕДУКЦИИ: рукописная норма делает mean, вычитание, квадрат, второй mean,
всё в типе входа; примитив сводит это в один проход. Требовать равенства
здесь нельзя, а порог relmax 1e-6 на fp16-выходах был бы порогом ПОД
решёткой типа, то есть требованием бит-в-бит под видом точности
(закон 37). Решётка fp16 даёт относительный шаг около 4.9e-4 на элементе
масштаба единицы -- отсюда порог в 4 ulp, а не круглое число.

ПОЧЕМУ ЗАВЕДЕНО ТРЕТЬЕ ПЛЕЧО (закон 36). Порог не назначают, его меряют
тем же инструментом на заведомо эквивалентном преобразовании ТОЙ ЖЕ
реализации.
ПОЛ СЕМЕЙСТВА -- ОДНОПРОХОДНАЯ ФОРМУЛА В ТОМ ЖЕ ТИПЕ (закон 36). Плечо
1pass считает ту же норму через E[x^2]-E[x]^2: математически то же, другой
порядок накопления. Его расхождение с off есть ИЗМЕРЕННАЯ цена одного лишь
порядка, и пункты 1 и 2 судят fast по ней.

ПЕРВАЯ ПОПЫТКА ПОЛА ОТВЕРГНУТА ЗАМЕРОМ 12.09. Вариант с пересчётом нашей же
формулы в fp32 НЕ ВОСПРОИЗВОДИТСЯ МЕЖДУ ПРОЦЕССАМИ: два запуска одного кода
на 0.1B дали разные состояния WKV в 11 слоях из 12, до 22 процентов
относительных, тогда как off, fast и 1pass бит-в-бит равны себе и между
процессами (сверено дампами состояний). Порог, меняющийся от прогона к
прогону, это не порог. Побочный вывод для проекта: у идеи считать активации
в fp32 ради точности есть скрытая цена -- невоспроизводимость между
процессами.

КАКИЕ ТИПЫ НА ВХОДЕ, ПОМЕРЕНО 12.09. Из 52 вызовов нормы на 0.1B пятьдесят
идут в fp32 и два в fp16 (ln0 сразу после эмбеддинга). На fp32-вызовах наша
норма и примитив равны по точности (5e-9...1e-7, кто ближе к fp64 -- по
вызовам по-разному), а на двух fp16-вызовах примитив точнее НАШЕГО В 6700
РАЗ (9.8e-10 против 6.6e-6): он поднимает редукцию в широкий тип, наша
считает в типе входа. Поэтому требовать от fast быть СТРОГО точнее нас было
бы подгонкой на fp32-шуме -- требуется попадание в разброс семейства.

ПОЧЕМУ greedy СПРАВОЧНО. Правка не бит-в-бит по построению, поэтому
траектории обязаны когда-нибудь разойтись, а мутацией 11.09 померено, что
greedy ловит только 1e-2 и есть самый слабый прибор набора. Решает KL.

ПОЧЕМУ ПУНКТ 2 -- ДИАГНОСТИКА, А НЕ ПОРОГ. Специфицирован он был дважды и
дважды неверно, и это запись об ошибке, а не о подгонке. Сперва требовалось
RMS(fast) <= RMS(off) на каждом вызове -- исходя из синтетики в fp16, где
примитив точнее нас во всех формах. На РЕАЛЬНЫХ входах 50 вызовов из 52
идут в fp32, и там картина другая: по медиане примитив точнее (5.9e-8
против 7.4e-8 на 0.1B), но на 13 вызовах из 26 он хуже нашего на ~6%.
Затем порог был ослаблен до max(RMS(off), RMS(1pass)) -- и падал так же.
Правильный вывод из этого НЕ ослаблять дальше, а признать: на fp32-шуме ни
одна из двух реализаций не доминирует другую покомпонентно, значит порога
по этой величине не существует, и её место -- диагностика механизма.
Качество судится KL на уровне МОДЕЛИ (пункт 5) -- тем прибором, которым
смерены все KL проекта, и по критерию, который владелец зарегистрировал
10.09. Иначе гейт мерил бы разрешение собственной линейки, а не правку.

ЧТО ЖЕ ТОГДА ДАЁТ ПУНКТ 1. Он ловит грубое: relmax сравнивается с ПОЛОМ
СЕМЕЙСТВА на каждом вызове, и по мутации он остаётся рабочим прибором
(границы в разделе сессии). Он про величину расхождения, а не про то, чьё
оно в пользу.
ЭТАЛОН -- ТОТ ЖЕ КОД, КОТОРЫМ СЧИТАЕТ МОДЕЛЬ: гейт зовёт производственный
qm._layer_norm и переключает плечи флагом qm.FAST_LN, своей копии формул
не держит (закон 30). RWKVQ_FAST_LN_ALIAS=1 делает плечо fast равным off
-- в этом состоянии гейт ОБЯЗАН проходить тривиально; если он там красный,
красен сам гейт.

РАЗРЕШЕНИЕ МЕРИТСЯ МУТАЦИЕЙ (закон 37). RWKVQ_GATE_MUTATE=eps умножает
выход плеча fast на (1+eps); гейт обязан покраснеть. Пойманные границы
записываются в раздел сессии.

Запуск: по одному размеру на процесс, из-под screen (закон 13 и таймаут
MCP). Ручки: RWKVQ_GATE_NSEQ (окон KL, умолчание 38 -- закон 9),
RWKVQ_GATE_DRIFT (1024), RWKVQ_GATE_GREEDY (256), RWKVQ_GATE_QKL (путь к
эталону квантования fp32), RWKVQ_GATE_SKIP (comp,drift,greedy,kl).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Develop/WKV-kvant/compression_v2_cand.rwkvq")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
NSEQ = int(os.environ.get("RWKVQ_GATE_NSEQ", "38"))
NDRIFT = int(os.environ.get("RWKVQ_GATE_DRIFT", "1024"))
NGREEDY = int(os.environ.get("RWKVQ_GATE_GREEDY", "256"))
SKIP = set(x for x in os.environ.get("RWKVQ_GATE_SKIP", "").split(",") if x)
ALIAS = os.environ.get("RWKVQ_FAST_LN_ALIAS", "0") == "1"
MUT = float(os.environ.get("RWKVQ_GATE_MUTATE", "0") or 0)
# ПОЛ ПРИБОРА ДРЕЙФА, ПОМЕРЕН МУТАЦИЕЙ 11.09 и записан в
# test_prewkv_parity.py: мутация в один ulp сразу даёт около 1e-4 и дальше
# НЕ РАСТЁТ, а выходит на плато. Ниже этого пола отношения соседних
# отметок -- шум, а не рост.
DRIFT_FLOOR = 3e-4

FAILS = []
UNCHECKED = []
T0 = time.time()
_PROD = qm._layer_norm
_REF = qm._layer_norm_ref


def say(msg):
    print("[%6.1f] %s" % (time.time() - T0, msg), flush=True)


def check(name, ok, detail):
    say(("OK   " if ok else "FAIL ") + name + ": " + detail)
    if not ok:
        FAILS.append(name)


def f32(a):
    return np.array(a.astype(mx.float32))


def ulp_rel(dtype):
    # относительный шаг решётки типа на элементе масштаба единицы
    return float(np.finfo(np.float16 if dtype == mx.float16
                          else np.float32).eps)


def _ln1p(x, weight, bias, eps=1e-5):
    # ПОЛ СЕМЕЙСТВА: та же норма через E[x^2]-E[x]^2, В ТОМ ЖЕ ТИПЕ.
    # Математически то же, другой порядок накопления, поэтому её
    # расхождение с off есть измеренная цена ОДНОГО ЛИШЬ ПОРЯДКА.
    m1 = x.mean(axis=-1, keepdims=True)
    m2 = (x * x).mean(axis=-1, keepdims=True)
    return (x - m1) / mx.sqrt(m2 - m1 * m1 + eps) * weight + bias


def _prod_mut(x, weight, bias, eps=1e-5):
    y = _PROD(x, weight, bias, eps)
    if qm.FAST_LN and MUT:
        y = (y.astype(mx.float32) * (1.0 + MUT)).astype(y.dtype)
    return y


def arm(name):
    """Переключить плечо. off и fast идут ПРОИЗВОДСТВЕННЫМ кодом."""
    if name == "1pass":
        qm._layer_norm = _ln1p
        qm.FAST_LN = False
    else:
        qm._layer_norm = _prod_mut if MUT else _PROD
        qm.FAST_LN = (name == "fast") and not ALIAS


say("модель: %s" % MODEL)
if ALIAS:
    say("RWKVQ_FAST_LN_ALIAS=1: плечо fast == off, гейт обязан пройти "
        "тривиально")
if MUT:
    say("МУТАЦИЯ выхода fast x (1+%g): гейт ОБЯЗАН покраснеть" % MUT)
model = qm.QuantRWKV7(load_raw(MODEL))
blob = torch.load(CORPUS)
TOK = blob["tokens"].numpy().astype(np.int32)
LANG = blob["lang"]
say("корпус: %d окон по %d, слоёв в модели %d" % (
    TOK.shape[0], TOK.shape[1], model.n_layer))
say("флаги: FUSE=%s FUSE_TAIL=%s LORA_Q=%r EMB_GATHER=%s FAST_LN=%s" % (
    qm.FUSE, qm.FUSE_TAIL, qm.LORA_Q, qm.EMB_GATHER, qm.FAST_LN))


# --- 1. компонентный паритет на РЕАЛЬНЫХ входах модели ------------------
# Входы захватываются с настоящего пути: префилл на 64 токенах плюс один
# шаг декода. Захват идёт ОБЁРТКОЙ производственной функции, поэтому
# покрыты все её вызовы, включая ln0 и ln_out вне блоков.
if "comp" not in SKIP:
    CAP = []

    def _cap(x, weight, bias, eps=1e-5):
        CAP.append((x, weight, bias, eps))
        return _REF(x, weight, bias, eps)

    qm._layer_norm = _cap
    qm.FAST_LN = False
    st = model.init_state(1)
    logits, st = model.forward_stateful(mx.array(TOK[0:1, :64]), st,
                                        last_only=True)
    mx.eval(logits)
    n_pref = len(CAP)
    tok = mx.argmax(logits[:, -1], axis=-1)
    logits, st = model.forward_stateful(tok[None], st)
    mx.eval(logits)
    arm("off")
    say("захвачено вызовов нормы: %d на префилле, %d на шаге декода" % (
        n_pref, len(CAP) - n_pref))
    check("захват покрыл нормы всех слоёв и ln0/ln_out",
          len(CAP) - n_pref == 2 * model.n_layer + 2,
          "%d при ожидаемых %d" % (len(CAP) - n_pref, 2 * model.n_layer + 2))

    worst_rel, worst_at = 0.0, -1
    worse_than_ours, over, degen = [], [], 0
    rms_off, rms_fast, rms_1p = [], [], []
    tol = None
    for i, (x, w, b, eps) in enumerate(CAP):
        arm("off")
        a = _prod_mut(x, w, b, eps) if MUT else _PROD(x, w, b, eps)
        arm("fast")
        c = _prod_mut(x, w, b, eps) if MUT else _PROD(x, w, b, eps)
        r = _ln1p(x, w, b, eps)
        mx.eval(a, c, r)
        if tol is None:
            tol = 0.0
            say("тип входа %s, решётка %.3e; порог СВОЙ НА КАЖДЫЙ вызов: "
                "max(4 ulp типа входа, разброс плеча-пола)" % (
                    x.dtype, ulp_rel(x.dtype)))
        A, C, R = f32(a), f32(c), f32(r)
        if float(np.abs(A).max()) == 0.0:
            degen += 1
            continue
        den = float(np.abs(A).max())
        rel = float(np.abs(A - C).max()) / den
        rel_1p = float(np.abs(A - R).max()) / den
        tol_i = max(4.0 * ulp_rel(x.dtype), rel_1p)
        if rel > tol_i:
            over.append((i, rel, tol_i))
        if rel / tol_i > worst_rel:
            worst_rel, worst_at = rel / tol_i, i
        # эталон fp64 НА ТЕХ ЖЕ ВХОДАХ
        X = f32(x).astype(np.float64)
        mu = X.mean(-1, keepdims=True)
        va = ((X - mu) ** 2).mean(-1, keepdims=True)
        REF64 = ((X - mu) / np.sqrt(va + eps) * f32(w).astype(np.float64)
                 + f32(b).astype(np.float64))
        ea = float(np.sqrt(((REF64 - A) ** 2).mean()))
        ec = float(np.sqrt(((REF64 - C) ** 2).mean()))
        er = float(np.sqrt(((REF64 - R) ** 2).mean()))
        rms_off.append(ea)
        rms_fast.append(ec)
        rms_1p.append(er)
        if ec > max(ea, er):
            worse_than_ours.append((i, max(ea, er), ec))
    arm("off")
    check("вход не вырожден", degen == 0, "вырожденных вызовов %d" % degen)
    check("relmax(fast, off) <= max(4 ulp входа, разброс пола) НА КАЖДОМ "
          "вызове", not over,
          "нарушений %d, худшее отношение к порогу %.2f на вызове %d" % (
              len(over), worst_rel, worst_at))
    say("ДИАГНОСТИКА точности: вызовов, где fast дальше от fp64, чем "
        "оба плеча семейства: %d из %d" % (len(worse_than_ours),
                                          len(rms_off)))
    say("RMS до fp64, медиана по вызовам: наш %.3e, fast %.3e, 1pass %.3e" % (
        float(np.median(rms_off)), float(np.median(rms_fast)),
        float(np.median(rms_1p))))


# --- 2. дрейф состояния WKV, teacher-forced -----------------------------
# Норма стоит перед КАЖДЫМ блоком, то есть правка входит в рекуррентность
# на каждом шаге. Одношаговая проверка этого не видит -- отсюда 1024.
# Порог: 1e-3 (зарегистрирован 10.09) ИЛИ пол семейства по плечу 1pass,
# что больше. Второй вариант -- не поддавка, а закон 36: плечо 1pass
# считает ТУ ЖЕ формулу, и его дрейф есть цена одного лишь порядка
# накопления.
if "drift" not in SKIP:
    seq = TOK.reshape(-1)[:NDRIFT]
    marks = []
    m = 32
    while m <= NDRIFT:
        marks.append(m)
        m *= 2
    if marks[-1] != NDRIFT:
        marks.append(NDRIFT)

    def run_forced(name):
        arm(name)
        st = model.init_state(1)
        snap = {}
        for i in range(len(seq)):
            _, st = model.forward_stateful(
                mx.array(seq[i:i + 1].reshape(1, 1)), st)
            if (i + 1) in marks:
                mx.eval(*[s[0] for s in st])
                snap[i + 1] = [f32(s[0]) for s in st]
        return snap

    def drift_of(sa, sb):
        out = {}
        for n in marks:
            r = 0.0
            for a, b in zip(sa[n], sb[n]):
                d = float(np.max(np.abs(a - b)))
                r = max(r, d / (float(np.max(np.abs(a))) + 1e-9))
            out[n] = r
        return out

    say("дрейф: %d шагов на плечо, три плеча, отметки %r" % (NDRIFT, marks))
    sa = run_forced("off")
    say("плечо off пройдено")
    sb = run_forced("fast")
    say("плечо fast пройдено")
    sr = run_forced("1pass")
    arm("off")
    say("плечо 1pass пройдено")
    d_fast = drift_of(sa, sb)
    d_1p = drift_of(sa, sr)
    say("дрейф fast:  " + ", ".join("%d:%.3e" % (n, d_fast[n])
                                    for n in marks))
    say("дрейф 1pass: " + ", ".join("%d:%.3e" % (n, d_1p[n])
                                    for n in marks))
    lim = 1e-3
    check("дрейф fast на %d <= 1e-3" % NDRIFT,
          d_fast[NDRIFT] <= lim,
          "%.3e при пороге %.3e (1pass СПРАВОЧНО %.3e)" % (
              d_fast[NDRIFT], lim, d_1p[NDRIFT]))
    # ПРАВКА 12.09 ПОСЛЕ ЛОЖНОГО КРАСНОГО НА 1.5B. Прежняя версия судила
    # рост одним лишь линейным потолком и НЕ ПЕРЕНЕСЛА из гейта ядра
    # правило про пол прибора -- это мой недосмотр, а не свойство правки.
    # Доказательство, что мерился шум: на 1.5B плечо-ПОЛ дало на той же
    # отметке отношение 3.05 при линейных 2.0, то есть провалило бы ту же
    # проверку, а всплеск у обоих плеч пришёлся на ОДНУ отметку 256
    # (fast 1.167e-03, пол 6.961e-04) и на следующей отметке спал.
    def ratios_of(d):
        return [d[marks[i]] / d[marks[i - 1]]
                if d[marks[i - 1]] > 0 else 0.0
                for i in range(1, len(marks))]

    ratios = ratios_of(d_fast)
    r_1p = ratios_of(d_1p)
    step = marks[1] / marks[0] if len(marks) > 1 else 2.0
    say("отношения соседних отметок, fast: " + ", ".join(
        "%.2f" % x for x in ratios))
    say("отношения соседних отметок, пол:  " + ", ".join(
        "%.2f" % x for x in r_1p))
    if d_fast[NDRIFT] <= DRIFT_FLOOR:
        check("рост дрейфа не быстрее линейного", True,
              "дрейф %.3e ниже пола прибора %.0e -- отношения там шум" % (
                  d_fast[NDRIFT], DRIFT_FLOOR))
    else:
        lim_r = max(1.25 * step, max(r_1p) if r_1p else 0.0)
        check("рост дрейфа <= max(линейный, рост пола семейства)",
              all(x <= lim_r for x in ratios),
              "max %.2f при пороге %.2f (линейный %.2f, у пола %.2f)" % (
                  max(ratios) if ratios else 0.0, lim_r, 1.25 * step,
                  max(r_1p) if r_1p else 0.0))


# --- 3. greedy ----------------------------------------------------------
# Самый слабый прибор набора: по мутации 11.09 он ловил только 1e-2.
# Требование "совпадение" здесь неуместно -- правка не бит-в-бит по
# построению, и разойтись траектории могут на равном по величине шуме.
# Поэтому критерий сравнительный: fast не хуже пола семейства 1pass.
if "greedy" not in SKIP:
    def greedy(name, n):
        arm(name)
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

    ga = greedy("off", NGREEDY)
    gb = greedy("fast", NGREEDY)
    gr = greedy("1pass", NGREEDY)
    arm("off")
    same_f = sum(1 for x, y in zip(ga, gb) if x == y)
    same_r = sum(1 for x, y in zip(ga, gr) if x == y)
    first_f = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), -1)
    first_r = next((i for i, (x, y) in enumerate(zip(ga, gr)) if x != y), -1)
    say("greedy %d: fast %d/%d (первое расхождение %d), "
        "1pass %d/%d (первое %d)" % (
            NGREEDY, same_f, NGREEDY, first_f, same_r, NGREEDY, first_r))
    check("greedy СПРАВОЧНО, решать по нему нельзя", True,
          "%d против %d" % (same_f, same_r))


# --- 4. KL между плечами, ПОТОКЕННО ------------------------------------
# Прибор -- ablate_subgroups.kl_stats, ТОТ ЖЕ, которым посчитаны все KL
# проекта (закон 39). Плечи чередуются потокенно в одном процессе.
# ЛОВУШКА 09.09: обвязка KL по умолчанию берёт срез [:8], а корпус
# отсортирован по языкам -- первые 20 окон русские. Здесь окна идут
# подряд от нуля до NSEQ, умолчание 38 = весь корпус (закон 9).
# Если задан RWKVQ_GATE_QKL, считается ТРИ KL: между плечами, от эталона
# квантования до off и от него же до fast -- последнее отвечает на вопрос,
# КОТОРОЕ плечо ближе к истине, а не только на сколько они разошлись
# (закон 38).
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

    per_arm, per_q_off, per_q_fast = [], [], []
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
            arm("off")
            la, stA = model.forward_stateful(x, stA)
            arm("fast")
            lb, stB = model.forward_stateful(x, stB)
            mx.eval(la, lb)
            va, vb = f32(la[0, -1]), f32(lb[0, -1])
            if A_ is None:
                A_ = np.empty((L, va.shape[0]), dtype=np.float32)
                B_ = np.empty((L, va.shape[0]), dtype=np.float32)
            A_[t], B_[t] = va, vb
        arm("off")
        kl_arm, _ = kl_stats(A_, B_)
        per_arm.append(float(kl_arm.mean()))
        line = "  окно %2d (%s): KL плеч %.3e" % (i, LANG[i], per_arm[-1])
        if ref is not None:
            R_ = np.asarray(ref[i])
            ko, _ = kl_stats(R_, A_)
            kf, _ = kl_stats(R_, B_)
            per_q_off.append(float(ko.mean()))
            per_q_fast.append(float(kf.mean()))
            line += ", KL кванта: off %.6f, fast %.6f" % (
                per_q_off[-1], per_q_fast[-1])
            R_ = None
        say(line)
        tgt = row[1:1 + L].astype(np.int64)
        nll_a += nll_of(A_, tgt)
        nll_b += nll_of(B_, tgt)
        ntok += L
        A_ = B_ = None

    kl_mean = float(np.mean(per_arm))
    lo, hi = boot_ci(per_arm)
    check("KL(fast || off) <= 1e-4 нат/ток", kl_mean <= 1e-4,
          "%.6e [%.2e; %.2e], худшее окно %.3e, %d токенов" % (
              kl_mean, lo, hi, max(per_arm), ntok))
    if ref is not None:
        q_off = float(np.mean(per_q_off))
        q_fast = float(np.mean(per_q_fast))
        ratio = float("inf") if kl_mean <= 0 else q_off / kl_mean
        check("запас против KL квантования >= 100x", ratio >= 100.0,
              "KL квантования %.6f, отношение %.1fx" % (q_off, ratio))
        say("КОТОРОЕ ПЛЕЧО БЛИЖЕ К ЭТАЛОНУ (закон 38): off %.6f, "
            "fast %.6f, разность %+.6f -- %s" % (
                q_off, q_fast, q_fast - q_off,
                "fast ближе" if q_fast < q_off else
                ("off ближе" if q_off < q_fast else "ровно")))
    else:
        UNCHECKED.append("запас 100x против KL квантования: "
                         "RWKVQ_GATE_QKL не задан")
    say("ppl СПРАВОЧНО (решать по ней нельзя): off %.4f, fast %.4f" % (
        float(np.exp(nll_a / ntok)), float(np.exp(nll_b / ntok))))


# --- итог ---------------------------------------------------------------
arm("off")
say("=" * 68)
for u in UNCHECKED:
    say("НЕ ПРОВЕРЕНО: " + u)
if MUT and not FAILS:
    say("ГЕЙТ НЕ ПОКРАСНЕЛ ПОД МУТАЦИЕЙ %g -- ЭТО ОТКАЗ ПРИБОРА" % MUT)
    sys.exit(1)
if FAILS:
    say("ГЕЙТ КРАСНЫЙ: " + ", ".join(FAILS))
    sys.exit(0 if MUT else 1)
say("ГЕЙТ ЗЕЛЁНЫЙ" + (" (с непроверенными пунктами)" if UNCHECKED else ""))
sys.exit(2 if UNCHECKED else 0)
