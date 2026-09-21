"""СЛАБОЕ МЕСТО МОДЕЛИ И ЛЕСТНИЦА БИТОВ (21.09). Один инструмент, fp32,
модель может лежать на НЕСКОЛЬКИХ картах (RWKVQ_DEVICES=cuda:0,cuda:1,...),
слои делятся подряд -- так 7.2B (28.8 ГБ) и 13.3B (53 ГБ) идут в fp32.

Режимы (env как у only_sweep_2109: RWKVQ_CKPT, RWKVQ_ACT_STATS, RWKVQ_KL_REF,
RWKVQ_CORPUS; окна RWKVQ_SWEEP_WIN, по умолчанию 3 ru / 3 en / 2 sr):
  proxy  <out>                 признак: пересчёт подслоя с одной квантованной
                               матрицей (ошибка в остаточном потоке / его норма)
                               x усиление G, измеренное шумом eps в той же точке.
                               RWKVQ_G_STRIDE=k: G мерится на слоях 0,1,2, каждом
                               k-м и трёх последних, между ними -- линейно.
  sweep  <out> [i/n]           истина: KL, когда квантована ОДНА матрица (LoRA
                               слоя одним плечом), остальное fp32.
  ladder <out> <proxy.json> [K]  кандидаты = верх-K признака + ВСЕ матрицы слоя 0
                               + ВСЕ матрицы последнего слоя + emb + head
                               (обе концевые зоны -- горячие по свипам 21.09;
                               признак недооценивает внутренние матрицы слоя 0).
                               Для каждого: KL(одна матрица в base) и KL(одна
                               матрица в base+1, ..., 6, bf16) + байты.
  joint  <out> <name> <json-переопределений>   KL модели целиком: базовый
                               конфиг + {ключ: бит}. Проверка набора правки.
Байты: params*(b+0.5)/8 при b<16 (0.5 бит/вес -- шкалы sb6: 12 бит на группу
32 и fp16-пара на суперблок 256; сверено с файлами: emb 2.9B 5->6 = +21.0 МБ),
2*params в bf16.
"""
import copy, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np
import torch
import torch.nn.functional as F
import ablate_sym_composite as comp
from rwkv_quant.calibration import fake_quant
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

MODE, OUT = sys.argv[1], sys.argv[2]
BASE = os.environ.get("RWKVQ_BASE_CFG", "compression_nol0")
DEVS = os.environ.get("RWKVQ_DEVICES", "cuda:0").split(",")
WIN = [int(x) for x in os.environ.get("RWKVQ_SWEEP_WIN", "0,6,13,20,24,28,30,35").split(",")]
EPS = float(os.environ.get("RWKVQ_PROXY_EPS", "0.01"))
GSTRIDE = int(os.environ.get("RWKVQ_G_STRIDE", "1"))
QGROUPS = ("proj", "cmix", "emb", "head")
LORA = ("w_lora", "a_lora", "v_lora", "g_lora")
cfg = comp.CONFIGS[BASE]()
C16 = QuantConfig()
t0 = time.time()

# ---------- модель на нескольких картах
M = RWKV7Ref(comp.CKPT, device=DEVS[0] if len(DEVS) == 1 else "cpu", dtype=torch.float32)
L, C = M.n_layer, M.n_embd
per = math.ceil(L / len(DEVS))
DEV = [DEVS[min(i // per, len(DEVS) - 1)] for i in range(L)]
D0, DL = DEVS[0], DEVS[-1]
if len(DEVS) > 1:
    mv = lambda t, d: t.to(d) if torch.is_tensor(t) else t          # noqa: E731
    M.emb_weight, M.ln0_w, M.ln0_b = mv(M.emb_weight, D0), mv(M.ln0_w, D0), mv(M.ln0_b, D0)
    M.head_weight, M.ln_out_w, M.ln_out_b = mv(M.head_weight, DL), mv(M.ln_out_w, DL), mv(M.ln_out_b, DL)
    for i in range(L):
        d = DEV[i]
        for lst in (M.ln1_w, M.ln1_b, M.ln2_w, M.ln2_b):
            lst[i] = mv(lst[i], d)
        for obj in (M.tmix[i], M.cmix[i]):
            for s in type(obj).__slots__:
                v = getattr(obj, s, None)
                if torch.is_tensor(v):
                    setattr(obj, s, v.to(d))
    torch.cuda.empty_cache()

blob = torch.load(comp.CORPUS)
data = blob["tokens"][WIN, :512].contiguous()[:, :-1].to(D0)
if MODE == "ref":
    # эталон fp32 38x511xV тем же многокарточным путём (13.3B: 53 ГБ fp32)
    WIN = list(range(blob["tokens"].shape[0]))
    data = blob["tokens"][WIN, :512].contiguous()[:, :-1].to(D0)
else:
    ref = np.load(os.environ["RWKVQ_KL_REF"], mmap_mode="r")
    LPREF = torch.log_softmax(torch.stack([torch.from_numpy(np.array(ref[w])) for w in WIN]).to(DL), -1)
print("[%s] загрузка %.0f с, слоёв %d, D %d, карт %d, окна %s" % (
    ",".join(DEVS), time.time() - t0, L, C, len(DEVS), WIN), flush=True)


def ln(x, w, b):
    return F.layer_norm(x.float(), (C,), w.float(), b.float())


@torch.no_grad()
def run(inject=None, cache=None):
    """Повтор RWKV7Ref.forward построчно + переходы между картами.
    Сверка с forward -- пол KL(__none__) против fp32-эталона."""
    x = ln(F.embedding(data, M.emb_weight), M.ln0_w, M.ln0_b)
    if inject: x = inject(("x0", -1), x)
    if cache is not None: cache["x0"] = x.clone()
    v_first, vf_on = torch.empty_like(x), {}
    for i in range(L):
        x = x.to(DEV[i])
        if cache is not None: cache[("in", i)] = x.clone()
        vf = v_first if i == 0 else vf_on.setdefault(DEV[i], v_first.to(DEV[i]))
        att, v_first = M._tmix_forward(ln(x, M.ln1_w[i], M.ln1_b[i]), vf, M.tmix[i], i, C16)
        if i == 0:
            vf_on = {DEV[0]: v_first}
            if cache is not None: cache["v_first"] = v_first.clone()
        x = x + att
        if inject: x = inject(("att", i), x)
        if cache is not None: cache[("mid", i)] = x.clone()
        x = x + M._cmix_forward(ln(x, M.ln2_w[i], M.ln2_b[i]), M.cmix[i], C16, i)
        if inject: x = inject(("ffn", i), x)
        if cache is not None: cache[("out", i)] = x.clone()
    h = ln(x.to(DL), M.ln_out_w, M.ln_out_b)
    if cache is not None: cache["h"] = h.clone()
    return h @ M.head_weight.T


def kl_of(lg):
    s = 0.0
    for j in range(lg.shape[0]):
        lp = LPREF[j].double(); lq = torch.log_softmax(lg[j].double(), -1)
        s += float((lp.exp() * (lp - lq)).sum(-1).mean())
    return s / lg.shape[0]


# ---------- точки квантования, базовая битность, байты
PTS = comp.quant_points(M)
ARMS = {}
for obj, attr, group, key in PTS:
    if group in QGROUPS:
        ARMS[key] = [(obj, attr, group, key)]
for obj, attr, group, key in PTS:
    if group in LORA and getattr(obj, attr) is not None:
        ARMS.setdefault("blocks.%d.lora" % int(key.split(".")[1]), []).append((obj, attr, group, key))


def base_bits(group, key, c=cfg):
    for pat, b in c.bits_overrides.items():          # подстрока, первое побеждает
        if pat in key:
            return b
    return c.bits[group]


def nbytes(params, b):
    return 2 * params if b >= 16 else params * (b + 0.5) / 8


def ladder_of(b):
    return list(range(b + 1, 7)) + [16] if b < 6 else ([16] if b < 16 else [])


FILE = 0.0
for obj, attr, group, key in PTS:
    w = getattr(obj, attr)
    if w is not None:
        FILE += nbytes(w.numel(), base_bits(group, key) if group in cfg.bits else 16)


def with_bits(sel, bits):
    """Подменить матрицы плеча: bits=None -- базовая битность конфига.
    Возвращает список для отката."""
    saved = []
    for obj, attr, group, key in sel:
        w = getattr(obj, attr)
        if w is None:
            continue
        c = cfg
        if bits is not None:
            c = copy.copy(cfg); c.bits_overrides = dict({key: bits}, **cfg.bits_overrides)
        q = fake_quant.q(w, group, c, key)
        if q is not w:
            saved.append((obj, attr, w)); setattr(obj, attr, q.to(w.dtype))
    return saved


def restore(saved):
    for obj, attr, w in saved:
        setattr(obj, attr, w)


def dump(res):
    json.dump(res, open(OUT + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(OUT + ".tmp", OUT)


if MODE == "ref":
    full = data
    mm = np.lib.format.open_memmap(OUT + ".tmp.npy", mode="w+", dtype=np.float32,
                                   shape=(full.shape[0], full.shape[1], M.head_weight.shape[0]))
    for b0 in range(0, full.shape[0], 8):
        data = full[b0:b0 + 8]
        lg = run()
        mm[b0:b0 + lg.shape[0]] = lg.float().cpu().numpy()
        print("  эталон %d/%d" % (b0 + lg.shape[0], full.shape[0]), flush=True)
    mm.flush(); del mm
    os.replace(OUT + ".tmp.npy", OUT)
    print("ГОТОВО ref за %.0f с" % (time.time() - t0), flush=True)
    sys.exit(0)

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res["_meta"] = dict(mode=MODE, ckpt=comp.CKPT, cfg=BASE, win=WIN, devs=DEVS, L=L, C=C,
                    file_bytes_est=FILE, eps=EPS, g_stride=GSTRIDE)
kl_none = kl_of(run())
res["_meta"]["kl_none"] = kl_none
print("пол KL(__none__) %.2e, файл (оценка) %.1f МБ" % (kl_none, FILE / 1e6), flush=True)
assert kl_none < 1e-5, "повтор forward разошёлся с эталоном"


def only_kl(arm, bits=None):
    saved = with_bits(ARMS[arm], bits)
    if not saved:
        return None
    kl = kl_of(run()); restore(saved)
    return kl


if MODE == "sweep":
    si, sn = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0/1").split("/"))
    rows = res.setdefault("rows", {})
    for j, arm in enumerate(ARMS):
        if j % sn != si or arm in rows:
            continue
        t1 = time.time(); kl = only_kl(arm)
        rows[arm] = {"kl": kl}
        print("%-34s KL %.6f  %.1f с" % (arm, kl or -1, time.time() - t1), flush=True); dump(res)

elif MODE == "proxy":
    cache = {}
    run(cache=cache)
    Gl = [0, 1, 2] + list(range(0, L, GSTRIDE)) + [L - 3, L - 2, L - 1]
    Gl = sorted(set(i for i in Gl if 0 <= i < L))
    G = res.setdefault("G", {})
    points = [("x0", -1)] + [(k, i) for i in Gl for k in ("att", "ffn")]
    t1 = time.time()
    for p in points:
        key = "%s:%d" % p
        if key in G:
            continue
        gen = torch.Generator(device=D0).manual_seed(0)
        def inj(q, x, p=p):
            if q != p:
                return x
            n = torch.randn(x.shape, generator=gen, device=D0).to(x.device)
            return x + n * (x.norm(dim=-1, keepdim=True) / n.norm(dim=-1, keepdim=True)) * EPS
        G[key] = kl_of(run(inject=inj)) / EPS ** 2
        dump(res)
    print("усиление: %d точек за %.0f с" % (len(points), time.time() - t1), flush=True)

    def g(kind, i):
        if kind == "x0":
            return G["x0:-1"]
        xs = [j for j in Gl]; ys = [G["%s:%d" % (kind, j)] for j in Gl]
        return float(np.interp(i, xs, ys))

    rows = res.setdefault("rows", {})

    @torch.no_grad()
    def local(arm):
        if arm == "emb.weight":
            return ln(F.embedding(data, M.emb_weight), M.ln0_w, M.ln0_b), cache["x0"], ("x0", -1)
        i = int(arm.split(".")[1])
        if ".ffn." in arm:
            xm = cache[("mid", i)]
            return M._cmix_forward(ln(xm, M.ln2_w[i], M.ln2_b[i]), M.cmix[i], C16, i), cache[("out", i)], ("ffn", i)
        xi = cache[("in", i)]
        vf = torch.empty_like(xi) if i == 0 else cache["v_first"].to(DEV[i])
        att, _ = M._tmix_forward(ln(xi, M.ln1_w[i], M.ln1_b[i]), vf, M.tmix[i], i, C16)
        return att, cache[("mid", i)], ("att", i)

    t2 = time.time()
    for arm, sel in ARMS.items():
        if arm == "head.weight":
            saved = with_bits(sel, None)
            rows[arm] = {"score": kl_of(cache["h"] @ M.head_weight.T), "point": "head"}
            restore(saved); continue
        b, xres, p = local(arm)
        saved = with_bits(sel, None)
        if not saved:
            continue
        a, _, _ = local(arm); restore(saved)
        rel2 = float(((a - b).norm(dim=-1) ** 2 / xres.norm(dim=-1) ** 2).mean())
        rows[arm] = {"rel2": rel2, "point": "%s:%d" % p, "G": g(*p), "score": rel2 * g(*p)}
    dump(res)
    print("локально: %d плеч за %.0f с" % (len(rows), time.time() - t2), flush=True)

elif MODE == "ladder":
    prox = json.load(open(sys.argv[3]))["rows"]
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    top = [k for k, _ in sorted(prox.items(), key=lambda kv: -kv[1]["score"]) if k in ARMS and not k.endswith(".lora")][:K]
    ends = [k for k in ARMS if not k.endswith(".lora") and (k.startswith("blocks.0.") or k.startswith("blocks.%d." % (L - 1)))]
    cand = list(dict.fromkeys(top + ends + ["emb.weight", "head.weight"]))
    lad = res.setdefault("ladder", {})
    for arm in cand:
        sel = ARMS[arm]; obj, attr, group, key = sel[0]
        b0 = base_bits(group, key)
        e = lad.setdefault(arm, {"group": group, "params": getattr(obj, attr).numel(), "base_bits": b0,
                                 "proxy_rank": (top.index(arm) + 1) if arm in top else None, "kl": {}})
        for b in [b0] + ladder_of(b0):
            if str(b) in e["kl"]:
                continue
            t1 = time.time()
            # bf16 одной матрицы при остальном fp32 -- это пол (0): вклад снят целиком
            e["kl"][str(b)] = 0.0 if b >= 16 else only_kl(arm, None if b == b0 else b)
            dump(res)
            print("%-32s %2d бит  KL %.6f  %.1f с" % (arm, b, e["kl"][str(b)], time.time() - t1), flush=True)

elif MODE == "joint":
    name, ovr = sys.argv[3], json.loads(sys.argv[4])
    saved, add = [], 0.0
    for obj, attr, group, key in PTS:
        w = getattr(obj, attr)
        if group not in cfg.bits or w is None:
            continue
        if key in ovr:
            add += nbytes(w.numel(), ovr[key]) - nbytes(w.numel(), base_bits(group, key))
        saved += with_bits([(obj, attr, group, key)], ovr.get(key))
    missing = [k for k in ovr if k not in {p[3] for p in PTS}]
    assert not missing, "ключей нет в модели: %s" % missing
    kl = kl_of(run()); restore(saved)
    res.setdefault("joint", {})[name] = {"kl": kl, "ovr": ovr, "add_bytes": add}
    dump(res)
    print("joint %-20s KL %.6f  +%.2f МБ (%.2f%% файла)" % (name, kl, add / 1e6, 100 * add / FILE), flush=True)
print("ГОТОВО %s за %.0f с" % (MODE, time.time() - t0), flush=True)
