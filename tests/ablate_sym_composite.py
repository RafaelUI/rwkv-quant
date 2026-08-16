"""Композит REDUCTION с Q6_K-раскладкой (`sym_aw`) на cmix и на proj.

ЧТО ЭТО ЗАКРЫВАЕТ. Изолированные замеры 12-13.08 (tests/ablate_sym_cmix.py)
дали на cmix -0.371 п.п. на 1.5B и -0.463 п.п. на 2.9B за +0.0625 бита/вес,
на proj -0.068 п.п. Закон 5 запрещает переносить изолированный выигрыш в
композит без замера: «комбинации конфигов непредсказуемы». Этот скрипт и
есть тот замер.

ПОЧЕМУ ПРЕДКВАНТОВАНИЕ, А НЕ ПУТЬ НА ЛЕТУ. `ablation.perplexity` зовёт
`fake_quant.q()` внутри forward, то есть переквантовывает КАЖДЫЙ батч.
Для изолированного замера это терпимо (квантуется одна группа), для
композита -- нет: 19 батчей x вся модель с грид-поиском. Кеш эту цену не
снимает, а на композите делает хуже по обеим осям сразу (закон 19:
набор ~2.8 ГБ не влезает в бюджет, кеш сдаётся и по дороге утаскивает
машину в своп на 4.4 ГБ).

Поэтому веса квантуются ОДИН раз и кладутся обратно в модель, а ppl
считается с QuantConfig() -- то есть по пути, где `q()` возвращает вход
без изменений. Память при этом не растёт вовсе: старая ссылка отпускается
сразу, живёт ровно одна копия весов.

Это оптимизация, а не другая схема, и она обязана быть бит-в-бит
тождественна пути на лету. Гейт: `--gate <ckpt>` строит ДВЕ модели на
маленьком чекпоинте, гоняет один батч по обоим путям и требует max|Δ| = 0
по логитам. Без зелёного гейта числам отсюда верить нельзя -- ровно тот
сорт ошибки, что в законе 15 (тихая подмена, которая не падает).

ЯЗЫКИ. ppl считается ПО ПОСЛЕДОВАТЕЛЬНОСТЯМ и разбивается на en/ru/sr:
сербский на 1.5B был канарейкой (5.52% против 1.64% en у REDUCTION до
small=16), и композитный эффект раскладки надо смотреть по нему отдельно,
а не только по ALL.

Один конфиг на процесс (законы 11 и 13). Результат дописывается в JSON.

    python tests/ablate_sym_composite.py --gate ~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth
    python tests/ablate_sym_composite.py bf16
    python tests/ablate_sym_composite.py reduction
    python tests/ablate_sym_composite.py reduction_sym_cmix
    python tests/ablate_sym_composite.py reduction_sym_cmix_proj
    python tests/ablate_sym_composite.py --report
"""
import copy
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rwkv_quant.calibration import fake_quant  # noqa: E402
from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.calibration.outlier_scan import GROUP_KEY_PATTERNS  # noqa: E402
from rwkv_quant.models.rwkv7_ref import RWKV7Ref  # noqa: E402
from rwkv_quant.presets import REDUCTION  # noqa: E402

CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth"))
CORPUS = os.environ.get("RWKVQ_CORPUS", os.path.expanduser(
    "~/Develop/WKV-kvant/eval_corpus_multiling.pt"))
ACT = os.environ.get("RWKVQ_ACT_STATS", "/tmp/act_stats_1p5b_ml.pt")
NSEQ = int(os.environ.get("RWKVQ_NSEQ", 38))
SEQLEN = int(os.environ.get("RWKVQ_SEQLEN", 512))
BATCH = int(os.environ.get("RWKVQ_BATCH", 2))
OUT = os.environ.get("RWKVQ_OUT", "/tmp/sym_composite.json")


# ---------------------------------------------------------------- конфиги

# ЗАМОРОЖЕННАЯ КОПИЯ ПРЕСЕТА ДО ПРАВКИ 16.08. Здесь она не из
# аккуратности, а потому что без неё ВСЕ ЗАПИСАННЫЕ В ТАБЛИЦАХ ЧИСЛА
# ПЕРЕСТАЮТ БЫТЬ ВОСПРОИЗВОДИМЫМИ.
#
# Конфиги ниже определены КАК ДЕЛЬТЫ поверх base_cfg(). Пока base_cfg
# читал живой `REDUCTION`, правка пресета молча переопределяла их все:
# после 16.08 `reduction_sym_head8` стал означать ровно новый пресет
# (proj@8, emb@8 уже внутри), и повторный прогон дал бы новые числа под
# старым именем. Поймано на замере ретривала: два разных конфига выдали
# ПОБАЙТОВО одинаковый результат, включая дрейф векторов 2.85e-04.
#
# Правило общее: имя конфига обозначает ЭКСПЕРИМЕНТ, чьё число записано,
# и обязано означать то же самое через полгода. Значит база эксперимента
# замораживается рядом с ним, а не берётся из изменяемого модуля.
LEGACY_REDUCTION = QuantConfig(
    proj=6, cmix=6, emb_head=6,
    w_lora=6, a_lora=6, v_lora=6, g_lora=8, small=16,
    outlier_fracs={},
    group_scale={"proj": 32, "cmix": 32, "emb_head": 32,
                 "w_lora": 64, "a_lora": 64, "v_lora": 64},
    group_scale_mode={"proj": "asym_sb6", "cmix": "asym_sb6_aw",
                      "emb_head": "asym_sb6_aw"},
    act_stats_path=None,
)


def base_cfg():
    """Пресет ДО правки 16.08, со статистикой ЭТОГО чекпоинта.

    Именно он -- база всех дельта-конфигов ниже, и именно на нём сняты
    все записанные числа. Нынешний пресет доступен отдельным именем
    `preset`; сравнивать их между собой можно, подменять один другим --
    нет.
    """
    cfg = copy.deepcopy(LEGACY_REDUCTION)
    cfg.act_stats_path = ACT
    return cfg


def preset_cfg():
    """НЫНЕШНИЙ пресет из presets.py, каким бы он ни был."""
    cfg = copy.deepcopy(REDUCTION)
    cfg.act_stats_path = ACT
    return cfg


def with_sym(cfg, *groups):
    """Перевести группы на Q6_K-раскладку: блок 16 вместо 32, режим sym_aw."""
    cfg = copy.deepcopy(cfg)
    cfg.group_scale = dict(cfg.group_scale, **{g: 16 for g in groups})
    cfg.group_scale_mode = dict(cfg.group_scale_mode,
                                **{g: "sym_aw" for g in groups})
    return cfg


def with_bits(cfg, **bits):
    """Сменить битность групп, не трогая всё остальное."""
    cfg = copy.deepcopy(cfg)
    cfg.bits = dict(cfg.bits, **bits)
    return cfg


def with_mode(cfg, group, mode):
    cfg = copy.deepcopy(cfg)
    cfg.group_scale_mode = dict(cfg.group_scale_mode, **{group: mode})
    return cfg


CONFIGS = {
    "bf16": lambda: QuantConfig(),
    # нынешний пресет как он есть в presets.py -- отдельным именем, чтобы
    # дельта-конфиги ниже остались привязаны к своей исторической базе
    "preset": preset_cfg,
    "reduction": base_cfg,
    "reduction_sym_cmix": lambda: with_sym(base_cfg(), "cmix"),
    "reduction_sym_cmix_proj": lambda: with_sym(base_cfg(), "cmix", "proj"),
    # КОНТРОЛЬ. В REDUCTION proj -- единственная группа без поиска и без
    # AW ("asym_sb6" plain, решение сессии 19.07-5). Значит переход
    # proj -> sym_aw меняет ТРИ вещи разом: раскладку, наличие поиска и
    # AW-взвешивание, и приписывать весь выигрыш раскладке нельзя.
    # Здесь proj получает поиск и AW в СТАРОЙ раскладке: разница с
    # reduction_sym_cmix -- цена поиска+AW, разница с
    # reduction_sym_cmix_proj -- цена собственно раскладки.
    "reduction_cmix_projaw": lambda: with_mode(
        with_sym(base_cfg(), "cmix"), "proj", "asym_sb6_aw"),
    # ПРАВКА ПРЕСЕТА, которую можно внести без единой строчки инженерии:
    # proj получает AW и поиск в НЫНЕШНЕЙ раскладке. Ноль байт разницы,
    # тот же упаковщик, тот же кернель, всё уже задеплоено. Мерится
    # отдельно от sym-cmix, потому что деплоить её можно и нужно
    # независимо от того, будет ли когда-нибудь написан упаковщик Q6_K.
    "reduction_projaw_only": lambda: with_mode(base_cfg(), "proj",
                                               "asym_sb6_aw"),
    # ПОЛНЫЙ КОМПОЗИТ ПРЕДЛАГАЕМОГО ПРЕСЕТА: cmix и proj в Q6_K на шести
    # битах ПЛЮС head в Q6_K на ВОСЬМИ. Изолированно (ablate_emb_head)
    # head -- единственная группа, где деградация значима и раскладкой
    # до конца не лечится: рычаг там БИТНОСТЬ. sym_aw@8 стоит 8.5625
    # бит/вес, то есть дешевле asym gw64@8 (9.000) при неразличимом
    # качестве, и обнуляет вклад группы на обоих масштабах.
    # Оценка по изолированным числам давала около +0.12% на 1.5B, но это
    # была ОЦЕНКА: закон 5 требует композитного замера прежде любой
    # правки пресетов, и вот он. emb НЕ трогается сознательно -- на 2.9B
    # он значим, но втрое дешевле head, а платить за него надо +43-52 МБ.
    "reduction_sym_head8": lambda: with_bits(
        with_sym(base_cfg(), "cmix", "proj", "head"), head=8),
    # ПЕРЕРАСПРЕДЕЛЕНИЕ БИТ, а не добавление. После head@8 весь остаток
    # бюджета сидит в proj: изолированно proj `sym_aw`@6 стоит +0.083%
    # на 1.5B, тогда как cmix в той же раскладке +0.005%, head@8 −0.002%,
    # emb +0.014% -- сумма сходится с композитом (+0.111%) почти целиком.
    # Значит вопрос «дать ли proj восемь бит» -- это вопрос про ВЕСЬ
    # остаток. Проверяется тем же упаковщиком: 8 бит уже умеет и он, и
    # кернель, ни строчки нового кода не нужно.
    "reduction_sym_proj8": lambda: with_bits(
        with_sym(base_cfg(), "cmix", "proj", "head"), proj=8, head=8),
    # ТРАТА ОСТАВШЕГОСЯ ЗАПАСА (16.08). Функция полезности REDUCTION
    # уточнена владельцем: борьбы за мегабайт там нет вовсе, требование --
    # "в 2-2.5 раза меньше bf16 при сопоставимом качестве". Для 1.5B это
    # 1222-1528 МБ при bf16 3055.3, то есть у кандидата (1299.2) остаётся
    # около 228 МБ запаса, а у proj8 (1399.8) -- 128.
    #
    # emb добавляется сюда потому, что по ppl он бесплатен, а по KL несёт
    # 24-26% композитной ошибки при 5.5-8.4% файла на ОБОИХ масштабах --
    # то есть два инструмента дают разный ответ, и разрешить спор можно
    # только замером обоих на одном и том же конфиге. Для векторных
    # моделей (ради которых REDUCTION и существует) верность
    # РАСПРЕДЕЛЕНИЯ важнее вероятности одного токена, поэтому ставка на
    # KL здесь не академическая.
    "reduction_sym_emb8": lambda: with_bits(
        with_sym(base_cfg(), "cmix", "proj", "head", "emb"),
        head=8, emb=8),
    "reduction_sym_proj8_emb8": lambda: with_bits(
        with_sym(base_cfg(), "cmix", "proj", "head", "emb"),
        proj=8, head=8, emb=8),
}


# ------------------------------------------------------------ размер файла

def bits_per_weight(group, cfg):
    """Бит/вес группы по её схеме. Формулы сверены с probe_schema_cost.py
    на реальном тензоре: sb6@6 = 6.500, asym gw64 = 9.000, rtn@8 = 8.021."""
    bits = cfg.bits[group]
    if bits >= 16:
        return 16.0
    gs = cfg.group_scale.get(group)
    mode = cfg.group_scale_mode.get(group, "asym")
    if not gs:
        # per-row RTN: суб-байтовая упаковка только при bits<=4 (_make_qt),
        # выше -- целый байт на код плюс fp32 scale на строку
        return (bits if bits <= 4 else 8) + 32 / 512
    if mode.startswith("sym"):
        sb = max(1, 256 // gs)
        return bits + 8 / gs + 16 / (gs * sb)
    if mode.startswith("asym_sb6"):
        return bits + (6 + 6) / gs + (16 + 16) / (gs * 8)
    # asym gw64: fp32 scale и min на блок, а КОДЫ ЛЕЖАТ В uint8 --
    # суб-байтовая упаковка есть только при bits<=4 (_make_qt). Отсюда
    # измеренные probe_schema_cost 9.000 бит/вес при bits=5, 6 И 8
    # одинаково: битность в этом контейнере не влияет на размер вовсе.
    # Считать её как `bits + 64/gs` -- занизить w/a/v_lora на четверть.
    return (bits if bits <= 4 else 8) + (32 + 32) / gs


def group_of(key):
    for g, pats in GROUP_KEY_PATTERNS.items():
        if any(key.endswith(p) or p in key for p in pats):
            return g
    return None


def size_mb(sd, cfg):
    """Оценка размера по группам. Это оценка ФОРМУЛОЙ, а не байты файла:
    реальный writer добавляет манифест и выравнивание (доли процента)."""
    total, per_group = 0.0, {}
    for k, t in sd.items():
        g = group_of(k)
        bpw = bits_per_weight(g, cfg) if (g and t.dim() == 2) else 16.0
        b = t.numel() * bpw / 8
        total += b
        per_group[g or "dense"] = per_group.get(g or "dense", 0.0) + b / 1e6
    return total / 1e6, per_group


# ------------------------------------------------------- предквантование

def quant_points(model):
    """(объект, атрибут, группа, ключ) для КАЖДОГО вызова q() в forward.

    Список повторяет models/rwkv7_ref.py построчно. Расхождение с ним --
    это замер не той схемы, поэтому он проверяется гейтом, а не глазами.
    """
    pts = [(model, "emb_weight", "emb", "emb.weight"),
           (model, "head_weight", "head", "head.weight")]
    for i in range(model.n_layer):
        t, c = model.tmix[i], model.cmix[i]
        pts += [
            (t, "r_proj", "proj", f"blocks.{i}.att.receptance.weight"),
            (t, "k_proj", "proj", f"blocks.{i}.att.key.weight"),
            (t, "v_proj", "proj", f"blocks.{i}.att.value.weight"),
            (t, "o_proj", "proj", f"blocks.{i}.att.output.weight"),
            (t, "w_lora_A", "w_lora", f"blocks.{i}.att.w1"),
            (t, "w_lora_B_w", "w_lora", f"blocks.{i}.att.w2"),
            (t, "a_lora_A", "a_lora", f"blocks.{i}.att.a1"),
            (t, "a_lora_B_w", "a_lora", f"blocks.{i}.att.a2"),
            (t, "g_lora_A", "g_lora", f"blocks.{i}.att.g1"),
            (t, "g_lora_B_w", "g_lora", f"blocks.{i}.att.g2"),
            (t, "k_k", "small", f"blocks.{i}.att.k_k"),
            (t, "k_a", "small", f"blocks.{i}.att.k_a"),
            (t, "r_k", "small", f"blocks.{i}.att.r_k"),
            (c, "key", "cmix", f"blocks.{i}.ffn.key.weight"),
            (c, "value", "cmix", f"blocks.{i}.ffn.value.weight"),
        ]
        # v_lora на нулевом слое в forward НЕ вызывается (v_first ещё не
        # определён), поэтому и здесь не квантуется -- иначе предквантованная
        # модель отличалась бы от пути на лету ровно на этих двух тензорах.
        if i > 0:
            pts += [(t, "v_lora_A", "v_lora", f"blocks.{i}.att.v1"),
                    (t, "v_lora_B_w", "v_lora", f"blocks.{i}.att.v2")]
    return pts


def prequantize(model, cfg, verbose=True):
    """Заменить веса квантованными ОДИН раз. Возвращает счётчик по группам."""
    done = {}
    t0 = time.time()
    for obj, attr, group, key in quant_points(model):
        w = getattr(obj, attr)
        if w is None:
            continue
        out = fake_quant.q(w, group, cfg, key)
        if out is not w:
            setattr(obj, attr, out.to(w.dtype))
            done[group] = done.get(group, 0) + 1
        del w, out
        if verbose and len(done) and sum(done.values()) % 64 == 0:
            print(f"  ... {sum(done.values())} тензоров, "
                  f"{time.time()-t0:.0f}s, {mem_line()}", flush=True)
    if verbose:
        print(f"  предквантование: {sum(done.values())} тензоров за "
              f"{time.time()-t0:.0f}s, по группам {done}", flush=True)
    return done


# ------------------------------------------------------------------ замер

def mem_line():
    """Системные цифры: RSS и метрики фреймворков для unified memory врут
    (закон 11), поэтому vm.swapusage и свободный процент из memory_pressure."""
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True).stdout.strip()
    used = sw.split("used =")[1].split()[0] if "used =" in sw else "?"
    mp = subprocess.run(["memory_pressure", "-Q"],
                        capture_output=True, text=True).stdout
    free = next((l.split(":")[1].strip() for l in mp.splitlines()
                 if "free percentage" in l), "?")
    return f"своп {used}, свободно {free}"


@torch.no_grad()
def nll_by_seq(model, data, batch=1):
    """(сумма NLL, число предсказаний) на КАЖДУЮ последовательность.

    NLL через logsumexp, а не log_softmax: логиты [B,T,65536], полная
    материализация log-вероятностей стоит ~0.8 ГБ транзиента на батч.
    """
    out = []
    for i in range(0, data.shape[0], batch):
        b = data[i:i + batch]
        logits = model.forward(b[:, :-1]).float()
        tgt = b[:, 1:]
        nll = (torch.logsumexp(logits, dim=-1)
               - logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
        for r in range(nll.shape[0]):
            out.append((float(nll[r].sum().item()), int(nll[r].numel())))
        del logits, nll
    return out


def ppl_from(pairs):
    s = sum(x[0] for x in pairs)
    n = sum(x[1] for x in pairs)
    if n == 0:
        return float("nan")
    mean = s / n
    # ppl = 1.0000 -- это не результат, а упавший command buffer (закон 21)
    if not (mean > 1e-6) or mean != mean or mean == float("inf"):
        raise RuntimeError(
            f"ppl вырождена (средний NLL = {mean!r}): почти наверняка MPS "
            f"исчерпал память и уронил command buffer. Смотрите stderr на "
            f"'Insufficient Memory'.")
    return float(torch.exp(torch.tensor(mean)))


# ----------------------------------------------------------------- гейт

def gate(ckpt):
    """Предквантование обязано быть бит-в-бит тождественно пути на лету.

    Две модели на маленьком чекпоинте, один батч, сравнение логитов.
    Гоняется на 0.1B: две копии 1.5B на 16 ГБ не нужны никому.
    """
    print(f"=== гейт предквантования на {os.path.basename(ckpt)} ===")
    d = torch.load(CORPUS)
    tok = d["tokens"] if isinstance(d, dict) else d
    data = tok[:2, :128].contiguous().to("mps")
    cfg = with_sym(base_cfg(), "cmix", "proj")

    m1 = RWKV7Ref(ckpt, device="mps", dtype=torch.bfloat16)
    fake_quant.cache_begin(0)          # без кеша: сравниваем сами схемы
    a = m1.forward(data, cfg).float().cpu()
    fake_quant.cache_end()
    del m1

    m2 = RWKV7Ref(ckpt, device="mps", dtype=torch.bfloat16)
    prequantize(m2, cfg)
    b = m2.forward(data).float().cpu()
    del m2

    dmax = float((a - b).abs().max())
    same = bool(torch.equal(a, b))
    print(f"max|Δ| логитов = {dmax:.3e}, бит-в-бит: {same}")
    print("ЗЕЛЁНЫЙ" if same else
          "КРАСНЫЙ -- предквантование обходит не те тензоры, числам верить нельзя")
    return 0 if same else 1


# --------------------------------------------------------------- отчёт

def load_doc():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {"ckpt": CKPT, "corpus": CORPUS, "act": ACT, "rows": {}}


def report(doc):
    rows = doc.get("rows", {})
    if not rows:
        print("нет данных")
        return
    bf = rows.get("bf16", {}).get("ppl")
    langs = doc.get("langs", [])
    uniq = sorted(set(langs))
    print(f"\nкомпозит, {doc.get('n_pred','?')} предсказаний, "
          f"{os.path.basename(doc.get('ckpt',''))}, bf16 ppl={bf}")
    # ALL смещён составом корпуса (ru 20 / en 9 / sr 9 последовательностей),
    # поэтому рядом печатается СРЕДНЕЕ ПО ЯЗЫКАМ -- равный вес каждому.
    # Правка, которая сильно улучшает русский и слегка портит два других,
    # по ALL выглядит победой, а по среднему -- разменом; решать надо по
    # обоим числам, а не по одному.
    head = f"\n{'конфиг':26s} {'МБ':>8s} {'ppl':>9s} {'Δ ALL':>8s} {'Δ ср.яз':>8s}"
    for l in uniq:
        head += f" {('Δ ' + l):>8s}"
    print(head)
    for name in CONFIGS:
        r = rows.get(name)
        if not r:
            continue
        d = f"{100*(r['ppl']-bf)/bf:+7.3f}%" if bf else "       -"
        deltas = []
        for l in uniq:
            pl, bl = r.get("by_lang", {}).get(l), \
                rows.get("bf16", {}).get("by_lang", {}).get(l)
            deltas.append(100 * (pl - bl) / bl if pl and bl else None)
        ok = [x for x in deltas if x is not None]
        mean = f"{sum(ok)/len(ok):+7.3f}%" if ok else "       -"
        line = f"{name:26s} {r.get('mb',0):8.1f} {r['ppl']:9.4f} {d:>8s} {mean:>8s}"
        for x in deltas:
            line += f" {x:+7.3f}%" if x is not None else f" {'-':>8s}"
        print(line)
    base, sym = rows.get("reduction"), rows.get("reduction_sym_cmix")
    if base and sym and bf:
        db = 100 * (base["ppl"] - bf) / bf
        ds = 100 * (sym["ppl"] - bf) / bf
        print(f"\ncmix в Q6_K, композитом: {ds-db:+.3f} п.п. при "
              f"{sym['mb']-base['mb']:+.1f} МБ")
        print(f"  изолированно было -0.371 п.п. (1.5B) / -0.463 (2.9B); "
              f"перенос в композит {'подтверждён' if ds < db else 'НЕ состоялся'}")
    pr = rows.get("reduction_sym_cmix_proj")
    if sym and pr and bf:
        print(f"proj сверх того: "
              f"{100*(pr['ppl']-sym['ppl'])/bf:+.3f} п.п. при "
              f"{pr['mb']-sym['mb']:+.1f} МБ")
    ctl = rows.get("reduction_cmix_projaw")
    if sym and pr and ctl and bf:
        aw = 100 * (ctl["ppl"] - sym["ppl"]) / bf
        lay = 100 * (pr["ppl"] - ctl["ppl"]) / bf
        print(f"\nразложение выигрыша на proj (контроль), по ALL:")
        print(f"  поиск + AW в старой раскладке: {aw:+.3f} п.п.")
        print(f"  собственно Q6_K-раскладка:     {lay:+.3f} п.п. (+{pr['mb']-ctl['mb']:.1f} МБ)")
        # Вывод НЕ делается по одному ALL: правка, которая берёт своё на
        # русском и отдаёт на сербском, по ALL выглядит выигрышем при
        # смещённом корпусе. Смотрим, улучшились ли ВСЕ языки.
        def per_lang(a, b):
            return {l: 100 * (b["by_lang"][l] - a["by_lang"][l])
                    / rows["bf16"]["by_lang"][l] for l in uniq
                    if l in a.get("by_lang", {}) and l in b.get("by_lang", {})}
        d_aw, d_lay = per_lang(sym, ctl), per_lang(ctl, pr)
        print(f"  по языкам, AW:       " +
              "  ".join(f"{l} {v:+.3f}" for l, v in d_aw.items()))
        print(f"  по языкам, раскладка:" +
              "  ".join(f"{l} {v:+.3f}" for l, v in d_lay.items()))
        both = all(v < 0 for v in d_lay.values())
        print("  -> " + ("раскладка улучшает ВСЕ языки" if both else
                         "раскладка улучшает не все языки") +
              "; AW " + ("улучшает все" if all(v < 0 for v in d_aw.values())
                         else "часть языков ПОРТИТ -- это размен, не правка"))


# ---------------------------------------------------------------- main

def main():
    a = sys.argv[1:]
    if not a or a[0] == "--report":
        report(load_doc())
        print(f"\nконфиги: {' '.join(CONFIGS)}")
        return 0
    if a[0] == "--sizes":
        # пересчитать МБ во всех строках без единого прогона: размер --
        # функция конфига и форм, а не результата замера
        doc = load_doc()
        sd = torch.load(CKPT, map_location="cpu", mmap=True)
        for name, r in doc.get("rows", {}).items():
            old = r.get("mb", 0)
            # per_group_mb ОБЯЗАН обновляться вместе с mb: пока он не
            # переписывался, итог был по новой формуле, а разбивка -- по
            # прежней (та занижала asym gw64), и сравнение групп между
            # строками давало несуществующие дельты
            r["mb"], r["per_group_mb"] = size_mb(sd, CONFIGS[name]())
            print(f"{name:26s} {old:8.1f} -> {r['mb']:8.1f} МБ")
        with open(OUT, "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
        return 0
    if a[0] == "--gate":
        return gate(a[1] if len(a) > 1 else os.path.expanduser(
            "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
    name = a[0]
    if name not in CONFIGS:
        raise SystemExit(f"неизвестный конфиг {name}; есть: {list(CONFIGS)}")
    cfg = CONFIGS[name]()
    if name != "bf16" and not os.path.exists(ACT):
        raise SystemExit(f"нет статистики активаций {ACT}: AW-режимы молча "
                         f"выродятся в _search, и замер будет не тот (закон 15)")

    print(f"=== {os.path.basename(CKPT)} / {name} ===", flush=True)
    print(f"старт: {mem_line()}", flush=True)
    blob = torch.load(CORPUS)
    tok = blob["tokens"] if isinstance(blob, dict) else blob
    langs = list(blob.get("lang", []))[:NSEQ] if isinstance(blob, dict) else []
    data = tok[:NSEQ, :SEQLEN].contiguous().to("mps")

    model = RWKV7Ref(CKPT, device="mps", dtype=torch.bfloat16)
    t0 = time.time()
    if name != "bf16":
        prequantize(model, cfg)
        print(f"после кванта: {mem_line()}", flush=True)

    pairs = nll_by_seq(model, data, batch=BATCH)
    p = ppl_from(pairs)
    by_lang = {}
    for l in sorted(set(langs)):
        sel = [pr for pr, ll in zip(pairs, langs) if ll == l]
        by_lang[l] = ppl_from(sel)

    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    mb, per_group = size_mb(sd, cfg)

    doc = load_doc()
    doc["langs"] = langs
    doc["n_pred"] = sum(x[1] for x in pairs)
    doc["rows"][name] = {"ppl": p, "mb": mb, "by_lang": by_lang,
                         "per_group_mb": per_group,
                         "seconds": round(time.time() - t0)}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"{name}: ppl={p:.4f}  {mb:.1f} МБ  [{time.time()-t0:.0f}s]", flush=True)
    print(f"по языкам: " + "  ".join(f"{k} {v:.4f}" for k, v in by_lang.items()))
    print(f"финиш: {mem_line()}", flush=True)
    report(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
