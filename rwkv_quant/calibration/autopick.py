"""ПРАВИЛО «СЛАБОЕ МЕСТО -> БИТ» В ОБЕ СТОРОНЫ (autopick), 22.09.

Вход -- ИЗМЕРЕНИЕ (dict, см. measure): для каждой квантуемой матрицы KL модели,
у которой квантована ТОЛЬКО она (битность пресета), остальное -- в типе счёта
эталона; для матриц, у которых есть реализуемый шаг вниз, -- ещё KL на бит ниже.
Плюс KL всего пресета (kl_all) и оценка байт файла пресета (file_bytes).

Выбор (select):
  ВВЕРХ -- жадно, по e = (снятая доля KL) / (добавленная доля байт), пока e >= tau.
    Уровни выше пресета НЕ мерятся: KL(b+j) = KL(b) * r^j, bf16 -> 0. Закон уровней
    измерен на 6 масштабах 0.1B-13.3B (500+ шагов): r = 0.315 для proj/cmix
    (медианы 0.315/0.313/0.317 по типам), 0.41 для головы. На полной истине выбор
    по закону уровней дал тот же КПД (%KL/МБ), что и выбор по истинным лестницам,
    на всех пяти проверенных масштабах (NEXT_SESSION, 22.09 ночь).
  ВНИЗ -- для НЕподнятых матриц шаг b -> b-1, если он РЕАЛИЗУЕМ (b-1 >= 4: у
    asym_sb6 нет 3 бит; шаги на 3 бита при e<5 дают <1% файла -- раскладку не
    заводим) и e_down = (добавленная доля KL)/(снятая доля байт) < tau.
    emb вниз не идёт (решение владельца: emb 6 всегда).
Один tau с обеих сторон = одна цена байта: вверх -- где байт окупается сильнее,
вниз -- где слабее. Байты: params*(b+0.5)/8 при b<16 (0.5 бит/вес -- шкалы sb6),
2*params в bf16."""

R_LEVEL = {"proj": 0.315, "cmix": 0.315, "head": 0.41, "emb": 0.27}
TOP_BITS = 6       # выше -- только bf16
MIN_REAL = 4       # нижняя реализуемая битность (asym_sb6_aw: 4/5/6)
NO_DOWN = ("emb.weight",)


def nbytes(params, bits):
    return 2 * params if bits >= 16 else params * (bits + 0.5) / 8


def _ladder(arm):
    b0, p = arm["bits"], arm["params"]
    k0 = arm["kl"][str(b0)]
    r = R_LEVEL[arm["group"]]
    lev = [(b, k0 * r ** (b - b0), nbytes(p, b)) for b in range(b0, TOP_BITS + 1)]
    return lev + [(16, 0.0, nbytes(p, 16))]


def select(measure, tau, down=True):
    """-> (bits_overrides {ключ: бит}, отчёт dict). Ключи -- точные ключи state_dict;
    вызывающий обязан ставить их ПЕРВЫМИ в bits_overrides (подстрока, первое побеждает)."""
    KL_ALL, FILE = measure["kl_all"], measure["file_bytes"]
    assert KL_ALL > 0 and FILE > 0
    arms = {k: a for k, a in measure["arms"].items() if a["bits"] < 16}
    lv = {k: _ladder(a) for k, a in arms.items()}
    cur = {k: 0 for k in lv}
    up_log = []
    while True:
        best = None
        for k, lev in lv.items():
            b0, k0, y0 = lev[cur[k]]
            for j in range(cur[k] + 1, len(lev)):
                b1, k1, y1 = lev[j]
                e = ((k0 - k1) / KL_ALL) / ((y1 - y0) / FILE)
                if best is None or e > best[0]:
                    best = (e, k, j)
        if best is None or best[0] < tau:
            break
        e, k, j = best
        up_log.append((k, lv[k][cur[k]][0], lv[k][j][0], e))
        cur[k] = j
    ovr = {k: lv[k][cur[k]][0] for k in lv if cur[k]}
    d_kl = sum(lv[k][0][1] - lv[k][cur[k]][1] for k in ovr)
    d_up = sum(lv[k][cur[k]][2] - lv[k][0][2] for k in ovr)
    dn_log, k_dn, y_dn = [], 0.0, 0.0
    if down:
        for k, a in arms.items():
            b0 = a["bits"]
            if k in ovr or k in NO_DOWN or b0 - 1 < MIN_REAL or str(b0 - 1) not in a["kl"]:
                continue
            dk = a["kl"][str(b0 - 1)] - a["kl"][str(b0)]
            dy = nbytes(a["params"], b0) - nbytes(a["params"], b0 - 1)
            e = (dk / KL_ALL) / (dy / FILE)
            if e < tau:
                ovr[k] = b0 - 1; dn_log.append((k, b0, b0 - 1, e)); k_dn += dk; y_dn += dy
    ram = d_up - y_dn - sum(lv[k][cur[k]][2] - lv[k][0][2] for k in ovr if k == "emb.weight" and cur.get(k))
    rep = dict(tau=tau, n_up=len(up_log), n_down=len(dn_log),
               kl_pred=(k_dn - d_kl) / KL_ALL, bytes=d_up - y_dn, ram_bytes=ram,
               up=up_log, down=dn_log)
    return ovr, rep



def select_budget(measure, budget, down=True, tau_lo=1.0, tau_hi=100.0, iters=60):
    """Выбор при БЮДЖЕТЕ БАЙТ вместо цены (решение владельца 24.09: бюджет -- ДОЛЯ ФАЙЛА).
    budget -- допустимое изменение байт файла долей от file_bytes (0.005 = +0.5%; 0 --
    байт-нейтрально; отрицательный -- обязательная экономия).

    select(tau) -- лагранжево решение при цене байта tau: вверх, где e >= tau, вниз, где
    e < tau. Байты select(tau) не возрастают с tau (подъёмы -- префикс одной и той же жадной
    последовательности, спуски -- растущее множество), и KL-выигрыш тоже. Поэтому лучший
    выбор в бюджете -- select при НАИМЕНЬШЕМ tau, чьи байты <= бюджета: ищется бисекцией
    по log tau. Если бюджет не достижим даже при tau_hi -- возвращается выбор при tau_hi
    с rep["feasible"] = False (вызывающий решает); если достижим уже при tau_lo -- tau_lo.
    -> (bits_overrides, отчёт select + tau, budget, bytes_frac, feasible)."""
    import math
    FILE = measure["file_bytes"]
    lim = budget * FILE

    def at(t):
        o, r = select(measure, t, down=down)
        return o, r

    o, r = at(tau_lo)
    if r["bytes"] <= lim:
        best = (tau_lo, o, r)
    else:
        o_hi, r_hi = at(tau_hi)
        if r_hi["bytes"] > lim:
            r_hi.update(tau=tau_hi, budget=budget, bytes_frac=r_hi["bytes"] / FILE, feasible=False)
            return o_hi, r_hi
        lo, hi, best = math.log(tau_lo), math.log(tau_hi), (tau_hi, o_hi, r_hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            om, rm = at(math.exp(mid))
            if rm["bytes"] <= lim:
                hi, best = mid, (math.exp(mid), om, rm)
            else:
                lo = mid
    t, o, r = best
    r.update(tau=t, budget=budget, bytes_frac=r["bytes"] / FILE, feasible=True)
    return o, r


def reweight(measure, weights):
    """Измерение с ВЕСАМИ ЯЗЫКОВ в цели выбора (24.09): KL плеча и KL пресета заменяются на
    sum_l w_l * (средний KL по окнам языка l). weights -- {язык: вес}, нормируются к сумме 1;
    язык окон без веса получает 0; вес языка, которого нет в окнах, -- ошибка. Нужны KL по
    окнам (measure v3). Результат -- обычное измерение: select/select_budget без изменений.
    При весах, пропорциональных числу окон, совпадает с исходным (до округления)."""
    import copy as _cp
    if "kl_all_w" not in measure:
        raise ValueError("в измерении нет KL по окнам (нужна версия >= 3): пересчитать measure")
    langs = measure["langs"]
    miss = [l for l in weights if l not in langs]
    if miss:
        raise ValueError("веса для языков без окон: %s (окна: %s)" % (miss, sorted(set(langs))))
    tot = float(sum(weights.values()))
    assert tot > 0
    idx = {l: [j for j, x in enumerate(langs) if x == l] for l in weights}

    def agg(xs):
        return sum(weights[l] / tot * _mean([xs[j] for j in idx[l]]) for l in weights if weights[l])

    out = _cp.deepcopy(measure)
    out["kl_all"] = agg(measure["kl_all_w"])
    for k, a in out["arms"].items():
        a["kl"] = {b: agg(v) for b, v in a["klw"].items()}
    out["weights"] = {l: weights[l] / tot for l in weights}
    return out

# =====================================================================================
# ИЗМЕРЕНИЕ (measure). Прибор -- RWKV7Ref: веса хранятся в bf16, счёт в fp32 (численно
# та же fp32-модель, гейт tests/test_ref_storage_dtype.py). Эталон -- тот же прибор в
# том же процессе без квантования: пол нулевой по построению и проверяется.
# Цена: 1 проход на матрицу на битности пресета + 1 на реализуемый шаг вниз, но проход
# ВОЗОБНОВЛЯЕТСЯ с чистого остаточного потока на входе слоя матрицы (слои до неё
# квантование не меняет) -- в среднем вдвое дешевле полного, результат побитно тот же
# (гейт tests/test_autopick_measure.py). Память: bf16-веса + один остаточный поток +
# скрытое состояние эталона; логиты считаются по одному окну.
# =====================================================================================
import copy as _copy
import hashlib as _hashlib
import json as _json
import os as _os
import re as _re
import time as _time

MEASURE_VERSION = 3   # v3 (24.09): KL по каждому окну (klw, kl_all_w) -- для reweight
MEASURE_CACHE = _os.path.expanduser("~/.cache/rwkv-quant/measure")
SEQ_LEN = 512
# КВОТА ОКОН ПО ЯЗЫКАМ -- это ЦЕЛЕВАЯ функция выбора (чью деградацию правило снижает).
# Решение владельца 24.09: ШИРОКАЯ квота с весами пропорционально окнам (wprop). На g1j 1.5B
# при +0.5% файла прицел в язык этому языку почти не давал (только en: -16.5% против -16.3%),
# а ручные приоритеты (en .4/code .35) проиграли почти везде; 14 окон -- меньше шум выбора
# (половинки по 7 окон совпадают на 78%). Было (22.09): en3/code2/ru1/sr1/zh1.
QUOTA = (("en", 4), ("code", 4), ("ru", 2), ("sr", 2), ("zh", 2))
N_WINDOWS = sum(n for _, n in QUOTA)
_SR = set("јљњћђџЈЉЊЋЂЏ")


def _bits_of(cfg, group, key):
    for pat, b in cfg.bits_overrides.items():      # подстрока, первое побеждает (как writer)
        if pat in key:
            return b
    return cfg.bits[group]


def _points(M):
    """(объект, атрибут, группа, ключ) квантуемых матриц proj/cmix/emb/head. Только world-имена:
    ключи обязаны совпадать с ключами state_dict, по которым writer ставит bits_overrides."""
    if M.naming != "world":
        raise NotImplementedError("autopick.measure: только чекпоинты с world-именами")
    pts = [(M, "emb_weight", "emb", "emb.weight")]
    for i, (t, c) in enumerate(zip(M.tmix, M.cmix)):
        for attr, name in (("r_proj", "receptance"), ("k_proj", "key"), ("v_proj", "value"), ("o_proj", "output")):
            pts.append((t, attr, "proj", "blocks.%d.att.%s.weight" % (i, name)))
        pts.append((c, "key", "cmix", "blocks.%d.ffn.key.weight" % i))
        pts.append((c, "value", "cmix", "blocks.%d.ffn.value.weight" % i))
    pts.append((M, "head_weight", "head", "head.weight"))
    return pts


def _file_points(M):
    """ВСЕ точки q() в forward (как tests/ablate_sym_composite.quant_points): знаменатель
    «доли файла» в e обязан совпадать с тем, на котором откалиброван tau (22.09)."""
    pts = list(_points(M))
    for i, t in enumerate(M.tmix):
        for attr, g, k in (("w_lora_A", "w_lora", "w1"), ("w_lora_B_w", "w_lora", "w2"),
                           ("a_lora_A", "a_lora", "a1"), ("a_lora_B_w", "a_lora", "a2"),
                           ("g_lora_A", "g_lora", "g1"), ("g_lora_B_w", "g_lora", "g2"),
                           ("k_k", "small", "k_k"), ("k_a", "small", "k_a"), ("r_k", "small", "r_k")) + \
                ((("v_lora_A", "v_lora", "v1"), ("v_lora_B_w", "v_lora", "v2")) if i > 0 else ()):
            pts.append((t, attr, g, "blocks.%d.att.%s" % (i, k)))
    return pts


def _lang(text):
    """Грубая, но детерминированная метка чанка корпуса: code / zh / sr / ru / en."""
    t = text[:4000]
    if _re.search(r"^\s*(import |from \S+ import |def |class |func |struct |let |var |#include|//)", t, _re.M):
        return "code"
    cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    cyr = sum(1 for ch in t if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    if cjk > max(cyr, lat):
        return "zh"
    if cyr > lat:
        return "sr" if any(ch in _SR for ch in t) else "ru"
    return "en"


def _pick_windows(encode, corpus_path, seq_len, quota=QUOTA):
    """Окна калибровочного корпуса по языковой квоте. Внутри языка -- по кругу между
    чанками (первое окно каждого чанка, затем вторые...), чтобы окна одного языка шли
    из разных текстов. -> (окна, метки)."""
    with open(corpus_path, encoding="utf-8") as _f:
        text = _f.read()
    chunks = [c.strip() for c in _re.split(r"—+ CHUNK —+", text) if c.strip()]
    by = {}
    for c in chunks:
        ids = encode(c)
        wins = [ids[w * seq_len:(w + 1) * seq_len] for w in range(len(ids) // seq_len)]
        if wins:
            by.setdefault(_lang(c), []).append(wins)
    out, labels = [], []
    for lang, n in quota:
        pool = by.get(lang, [])
        order = [w for d in range(max((len(x) for x in pool), default=0)) for x in pool if d < len(x)
                 for w in [x[d]]]
        if len(order) < n:
            raise RuntimeError("калибровочный корпус: для %r %d окон, нужно %d" % (lang, len(order), n))
        out += order[:n]; labels += [lang] * n
    return out, labels


def _ckpt_signature(ckpt_path, blocks=64, bs=1 << 16):
    """Размер + 64 блока по 64 КБ равномерно по файлу. Подпись act_stats (размер + первый
    мегабайт) для переноса измерения слаба: в zip-чекпоинте torch первый мегабайт --
    pickle и начало ОДНОГО тензора, дообучение, его не тронувшее, подпись не меняет."""
    h = _hashlib.sha256(); size = _os.path.getsize(ckpt_path); h.update(str(size).encode())
    with open(ckpt_path, "rb") as f:
        for j in range(blocks):
            f.seek(max(0, (size - bs) * j // max(1, blocks - 1))); h.update(f.read(bs))
    return h.hexdigest()[:16]


def _file_hash(path):
    h = _hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


_WIN_MEMO = {}


def _measure_windows(tokenizer, corpus_path, seq_len):
    """(окна, метки) измерения по квоте -- один раз на (словарь, корпус, длину, квоту) в процессе:
    их берут и подпись, и сам measure."""
    from . import act_stats as A
    key = (A._memo_key(tokenizer), corpus_path, seq_len, repr(QUOTA))
    hit = _WIN_MEMO.get(key)
    if hit is None:
        hit = _WIN_MEMO[key] = (tokenizer,) + tuple(_pick_windows(A._encoder(tokenizer), corpus_path, seq_len, QUOTA))
    return hit[1], hit[2]


def _tokens_term(tokenizer, corpus_path, seq_len):
    """Словарь в подписи измерения -- хеш токенов его окон (06.10; было type(tokenizer).__name__,
    та же ошибка, что в act_stats._signature)."""
    from . import act_stats as A
    return A.tokens_sha(_measure_windows(tokenizer, corpus_path, seq_len)[0])


def _signature(ckpt_path, cfg, corpus_path, n, seq_len, tokenizer):
    """Подпись = ключ кеша И паспорт переносимого измерения: чекпоинт (выборка по всему
    файлу), конфиг (содержимое), AW-статистика (СОДЕРЖИМОЕ, не путь -- путь на другой
    машине другой), корпус, словарь (хеш токенов окон), квота окон, версия прибора."""
    from . import act_stats as A
    h = _hashlib.sha256()
    h.update(_ckpt_signature(ckpt_path).encode())
    h.update(_file_hash(corpus_path).encode())
    h.update(_tokens_term(tokenizer, corpus_path, seq_len).encode())
    h.update(repr(QUOTA).encode())
    h.update(repr(sorted(cfg.bits.items())).encode())
    h.update(repr(list(cfg.bits_overrides.items())).encode())
    h.update(repr(sorted((cfg.group_scale or {}).items())).encode())
    h.update(repr(sorted((cfg.group_scale_mode or {}).items())).encode())
    sp = getattr(cfg, "act_stats_path", None)
    h.update((_file_hash(sp) if sp else "no-act-stats").encode())
    h.update(("v%d n%d T%d" % (MEASURE_VERSION, n, seq_len)).encode())
    return h.hexdigest()[:16]


class _Instrument:
    """Прибор: чистый проход с запоминанием остаточного потока слой за слоем и возобновление."""

    def __init__(self, M, data):
        import torch
        import torch.nn.functional as F
        from .group_config import QuantConfig
        self.M, self.data, self.torch, self.F = M, data, torch, F
        self.C16 = QuantConfig()
        self.kdt = torch.float64 if str(data.device).split(":")[0] in ("cpu", "cuda") else torch.float32
        self.head32 = M.head_weight.to(M.cdtype)

    def _ln(self, x, w, b):
        return self.F.layer_norm(x.float(), (self.M.n_embd,), w.float(), b.float()).to(x.dtype)

    def x0(self):
        M = self.M
        x = self.F.embedding(self.data, M._q(M.emb_weight, "emb", self.C16, "emb.weight"))
        return self._ln(x, M.ln0_w, M.ln0_b)

    def att(self, i, x, vf):
        M = self.M
        x, vf = x.to(M.layer_dev[i]), vf.to(M.layer_dev[i])   # разнесение по картам (no-op на одной)
        a, vf = M._tmix_forward(self._ln(x, M.ln1_w[i], M.ln1_b[i]), vf, M.tmix[i], i, self.C16)
        return x + a, vf

    def ffn(self, i, x):
        M = self.M
        x = x.to(M.layer_dev[i])
        return x + M._cmix_forward(self._ln(x, M.ln2_w[i], M.ln2_b[i]), M.cmix[i], self.C16, i)

    def run_from(self, i, stage, x, vf):
        """Досчитать от слоя i (stage 'att' -- с TMix, 'ffn' -- с CMix) до скрытого после ln_out.
        Та же последовательность операций, что RWKV7Ref.forward."""
        M = self.M
        if stage == "att":
            x, vf = self.att(i, x, vf)
        x = self.ffn(i, x)
        for j in range(i + 1, M.n_layer):
            x, vf = self.att(j, x, vf)
            x = self.ffn(j, x)
        return self._ln(x.to(M.devices[-1]), M.ln_out_w, M.ln_out_b)

    def klw(self, h, h_ref, head=None):
        """KL(эталон || проба), средний по токенам, ПО КАЖДОМУ окну (логиты целиком не держатся)."""
        torch = self.torch
        W = self.head32 if head is None else head
        out = []
        for j in range(h.shape[0]):
            lp = torch.log_softmax((h_ref[j] @ self.head32.T).to(self.kdt), -1)
            lq = torch.log_softmax((h[j] @ W.T).to(self.kdt), -1)
            out.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        return out

    def kl(self, h, h_ref, head=None):
        """Среднее klw по окнам (тот же порядок суммирования, что до v3)."""
        return _mean(self.klw(h, h_ref, head))


def _mean(xs):
    s = 0.0
    for v in xs:
        s += v
    return s / len(xs)


def measure(ckpt_path, cfg, tokenizer, device=None, seq_len=SEQ_LEN,
            corpus_path=None, cache=True, verbose=True):
    """Измерение для select(): KL одной матрицы на битности конфига (остальное -- тип счёта
    эталона) и на бит ниже, где шаг реализуем; KL всего конфига; байты. -> dict (кешируется
    в ~/.cache/rwkv-quant/measure по подписи чекпоинта, конфига, корпуса и окон).
    cfg -- КОНФИГ ПРЕСЕТА с проставленным act_stats_path (AW-режимы без статистики вырождаются)."""
    import torch
    from . import act_stats as A
    from . import fake_quant
    from ..models.rwkv7_ref import RWKV7Ref
    corpus_path = corpus_path or A.CORPUS
    n_windows = sum(n for _, n in QUOTA)
    sig = _signature(ckpt_path, cfg, corpus_path, n_windows, seq_len, tokenizer)
    path = _os.path.join(MEASURE_CACHE, "measure_%s.json" % sig)
    if cache and _os.path.exists(path):
        if verbose:
            print("[autopick] измерение из кеша %s" % path)
        with open(path) as _f:
            return _json.load(_f)
    device = device or _os.environ.get("RWKVQ_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = _time.time()
    wins, langs = _measure_windows(tokenizer, corpus_path, seq_len)   # квота -- на момент вызова
    # "cuda:1,cuda:2" -- слои разнесены по картам (24.09); данные -- на первой
    devs = device.split(",") if isinstance(device, str) and "," in device else device
    M = RWKV7Ref(ckpt_path, device=devs, dtype=torch.bfloat16, compute_dtype=torch.float32)
    data = torch.tensor(wins, dtype=torch.long)[:, :-1].contiguous()
    if int(data.max()) >= M.vocab_size:
        raise ValueError("токенизатор не от этого чекпоинта: id %d при vocab %d" % (int(data.max()), M.vocab_size))
    data = data.to(M.devices[0])
    I = _Instrument(M, data)
    pts = _points(M)
    file_bytes = 0.0
    for obj, attr, group, key in _file_points(M):
        file_bytes += nbytes(getattr(obj, attr).numel(), _bits_of(cfg, group, key))

    def quantized(obj, attr, group, key, bits):
        c = _copy.copy(cfg); c.bits_overrides = dict({key: bits}, **cfg.bits_overrides)
        w = getattr(obj, attr)
        return fake_quant.q(w.to(M.cdtype), group, c, key)

    with torch.no_grad():
        # чистый проход: эталон и остаточные потоки -- по ходу, без кеша всех слоёв
        x = I.x0()
        vf = torch.empty_like(x)
        # эталон нужен целиком до начала замеров: отдельный чистый проход
        h_ref = I.run_from(0, "att", x, vf)
        floor = I.kl(h_ref, h_ref)
        kl_all_cfg = None
        arms = {}
        by_layer = {}
        for p in pts:
            by_layer.setdefault(p[3].split(".")[1] if p[3].startswith("blocks.") else p[3], []).append(p)

        def one(p, bits, start):
            obj, attr, group, key = p
            w = getattr(obj, attr)
            setattr(obj, attr, quantized(obj, attr, group, key, bits))
            try:
                if key == "head.weight":
                    return I.klw(h_ref, h_ref, head=getattr(obj, attr))
                return I.klw(start(), h_ref)
            finally:
                setattr(obj, attr, w)

        def record(p, start):
            obj, attr, group, key = p
            b0 = _bits_of(cfg, group, key)
            if b0 >= 16:
                return
            e = arms[key] = dict(group=group, params=getattr(obj, attr).numel(), bits=b0, kl={}, klw={})
            for b in [b0] + ([b0 - 1] if key not in NO_DOWN and b0 - 1 >= MIN_REAL else []):
                e["klw"][str(b)] = one(p, b, start)
                e["kl"][str(b)] = _mean(e["klw"][str(b)])

        record(by_layer["emb.weight"][0], lambda: I.run_from(0, "att", I.x0(), torch.empty_like(x)))
        record(by_layer["head.weight"][0], None)
        xi, vfi = x, vf
        t1 = _time.time()
        for i in range(M.n_layer):
            for p in by_layer[str(i)]:
                if p[2] == "proj":
                    record(p, lambda i=i, xi=xi, vfi=vfi: I.run_from(i, "att", xi, vfi))
            xm, vf_next = I.att(i, xi, vfi)
            for p in by_layer[str(i)]:
                if p[2] == "cmix":
                    record(p, lambda i=i, xm=xm, vf_next=vf_next: I.run_from(i, "ffn", xm, vf_next))
            xi, vfi = I.ffn(i, xm), vf_next
            if verbose:
                el = _time.time() - t1
                print("[autopick] слой %d/%d, %.0f с (осталось ~%.0f с)"
                      % (i + 1, M.n_layer, el, el / sum(M.n_layer - j for j in range(i + 1)) *
                         sum(M.n_layer - j for j in range(i + 1, M.n_layer))), flush=True)
        h_chk = I._ln(xi, M.ln_out_w, M.ln_out_b)
        resume_exact = bool(torch.equal(h_chk, h_ref))
        # KL всего конфига тем же прибором
        h_all = M.forward(data, cfg=cfg, return_hidden=True)
        kl_all_w = I.klw(h_all, h_ref, head=M._q(M.head_weight, "head", cfg, "head.weight"))
        kl_all_cfg = _mean(kl_all_w)
    if not resume_exact:
        raise RuntimeError("послойный проход разошёлся с эталоном: прибор неисправен")
    if not (floor < 1e-9 and kl_all_cfg > 0 and all(v >= 0 for a in arms.values() for v in a["kl"].values())):
        raise RuntimeError("измерение неправдоподобно (пол %.3g, KL конфига %.3g): отказ вместо файла"
                           % (floor, kl_all_cfg))
    out = dict(version=MEASURE_VERSION, signature=sig, ckpt=_os.path.basename(ckpt_path),
               device=str(device), n_windows=n_windows, seq_len=seq_len, langs=langs, floor=floor,
               ckpt_sig=_ckpt_signature(ckpt_path),
               kl_all=kl_all_cfg, kl_all_w=kl_all_w, file_bytes=file_bytes, arms=arms, seconds=round(_time.time() - t0, 1))
    if cache:
        _os.makedirs(MEASURE_CACHE, exist_ok=True)
        with open(path + ".tmp", "w") as _f:
            _json.dump(out, _f, indent=1)
        _os.replace(path + ".tmp", path)
    if verbose:
        print("[autopick] измерение: %d матриц, KL конфига %.6f, %.0f с -> %s" % (len(arms), kl_all_cfg, out["seconds"], path))
    return out
