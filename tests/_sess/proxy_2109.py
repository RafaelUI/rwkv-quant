"""ПРИЗНАК ДЛЯ АВТОПОДБОРА (21.09): ошибка в остаточном потоке x усиление.

Раздел 7 статьи: три признака провалены (статистика активаций, диагональная
ошибка весов, локальная ошибка выхода с нормировкой на ||WX||). Здесь:

  1. ЛОКАЛЬНО. Входы каждого блока кешируются одним прогоном. Для матрицы
     слоя l пересчитывается ТОЛЬКО её подслой (tmix или cmix) с квантованной
     матрицей: d = Δ(вклад подслоя в остаточный поток), нормировка на сам
     остаточный поток ПОСЛЕ добавления (там же, куда идёт шум): rel2 = mean_t ||d_t||^2/||x_t||^2.
     Так учитываются и внутренние матрицы (r/k/v/LoRA/ffn.key) -- через
     настоящую нелинейность блока, а не через ||ΔW X||.
  2. УСИЛЕНИЕ. В каждую точку добавления (x0 после ln0, после tmix и после
     cmix каждого слоя) впрыскивается шум относительной нормы EPS на токен,
     G = KL/EPS^2. Мерится, а не выводится из числа оставшихся слоёв: свип
     2.9B показал U-образный вред (слой 0 И слои 27-31).
  3. score = rel2 * G(точки). head -- точно: KL от одного ΔH (дальше ничего).
Сверка: ранги против only_sweep_2109 (Spearman, попадание верхних).
Известное упрощение: v слоя 0 кормит v_first всех слоёв -- здесь не видно.

    RWKVQ_DEVICE=cuda:0 python proxy_2109.py <конфиг> <выход.json>  (env как у свипа)
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np
import torch
import torch.nn.functional as F
import ablate_sym_composite as comp
from rwkv_quant.calibration import fake_quant
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

name, out_path = sys.argv[1], sys.argv[2]
DEV = os.environ.get("RWKVQ_DEVICE", "cuda:0")
WIN = [int(x) for x in os.environ.get("RWKVQ_SWEEP_WIN", "0,6,13,20,24,28,30,35").split(",")]
EPS = [float(e) for e in os.environ.get("RWKVQ_PROXY_EPS", "0.01").split(",")]
SEED = 0
cfg = comp.CONFIGS[name]()
C16 = QuantConfig()
blob = torch.load(comp.CORPUS)
data = blob["tokens"][WIN, :512].contiguous().to(DEV)[:, :-1]
ref = np.load(os.environ["RWKVQ_KL_REF"], mmap_mode="r")
REF = torch.stack([torch.from_numpy(np.asarray(ref[w])) for w in WIN]).to(DEV)   # fp32
LPREF = torch.log_softmax(REF, -1)          # fp32; KL считается в fp64 по окну
del REF

t0 = time.time()
M = RWKV7Ref(comp.CKPT, device=DEV, dtype=torch.float32)
L, C = M.n_layer, M.n_embd


def ln(x, w, b):
    return F.layer_norm(x.float(), (C,), w.float(), b.float())


@torch.no_grad()
def run(inject=None, cache=None):
    """Повтор RWKV7Ref.forward построчно (сверяется с ним ниже, закон 15).
    inject(point, x) -> x; точки: ("x0",), ("att", i), ("ffn", i)."""
    x = ln(F.embedding(data, M.emb_weight), M.ln0_w, M.ln0_b)
    if inject: x = inject(("x0", -1), x)
    if cache is not None: cache["x0"] = x.clone()
    v_first = torch.empty_like(x)
    for i in range(L):
        if cache is not None: cache[("in", i)] = x.clone()
        att, v_first = M._tmix_forward(ln(x, M.ln1_w[i], M.ln1_b[i]), v_first, M.tmix[i], i, C16)
        if cache is not None and i == 0: cache["v_first"] = v_first.clone()
        x = x + att
        if inject: x = inject(("att", i), x)
        if cache is not None: cache[("mid", i)] = x.clone()
        x = x + M._cmix_forward(ln(x, M.ln2_w[i], M.ln2_b[i]), M.cmix[i], C16, i)
        if inject: x = inject(("ffn", i), x)
        if cache is not None: cache[("out", i)] = x.clone()
    h = ln(x, M.ln_out_w, M.ln_out_b)
    if cache is not None: cache["h"] = h.clone()
    return h @ M.head_weight.T


def kl_of(lg):
    s = 0.0
    for j in range(lg.shape[0]):
        lp = LPREF[j].double(); lq = torch.log_softmax(lg[j].double(), -1)
        s += float((lp.exp() * (lp - lq)).sum(-1).mean())
    return s / lg.shape[0]


cache = {}
base = run(cache=cache)
kl0 = kl_of(base)
chk = M.forward(data[:1])[0]
dev_fwd = float((chk - base[0]).abs().max())
print("[%s] загрузка+кеш %.0f с, KL(база) %.2e, |повтор-forward| %.2e" % (DEV, time.time() - t0, kl0, dev_fwd), flush=True)
assert dev_fwd < 1e-3 and kl0 < 1e-6, "повтор forward разошёлся с RWKV7Ref.forward"

# ---- 2. усиление G(точка)
res = {"_meta": dict(cfg=name, win=WIN, eps=EPS, ckpt=comp.CKPT, kl_base=kl0)}
G = {}
points = [("x0", -1)] + [(k, i) for i in range(L) for k in ("att", "ffn")]
t1 = time.time()
for eps in EPS:
    for p in points:
        gen = torch.Generator(device=DEV).manual_seed(SEED)
        def inj(q, x, p=p, eps=eps):
            if q != p:
                return x
            n = torch.randn(x.shape, generator=gen, device=DEV)
            n = n * (x.norm(dim=-1, keepdim=True) / n.norm(dim=-1, keepdim=True)) * eps
            return x + n
        G.setdefault("%s:%d" % p, {})[str(eps)] = kl_of(run(inject=inj)) / eps ** 2
print("усиление: %d точек x %d eps за %.0f с" % (len(points), len(EPS), time.time() - t1), flush=True)
res["G"] = G
g = lambda p: G[p][str(EPS[0])]

# ---- 1. локальная ошибка в остаточном потоке
pts = comp.quant_points(M)
by_key = {}
for obj, attr, group, key in pts:
    if group in ("proj", "cmix", "emb", "head"):
        by_key[key] = [(obj, attr, group, key)]
lora = {}
for obj, attr, group, key in pts:
    if group in ("w_lora", "a_lora", "v_lora", "g_lora"):
        lora.setdefault("blocks.%d.lora" % int(key.split(".")[1]), []).append((obj, attr, group, key))
by_key.update(lora)


@torch.no_grad()
def local(arm):
    if arm == "emb.weight":
        x = ln(F.embedding(data, M.emb_weight), M.ln0_w, M.ln0_b)
        return x, cache["x0"], "x0:-1"
    if arm == "head.weight":
        return None, None, None
    i = int(arm.split(".")[1])
    if ".ffn." in arm:
        xm = cache[("mid", i)]
        d = M._cmix_forward(ln(xm, M.ln2_w[i], M.ln2_b[i]), M.cmix[i], C16, i)
        return d, cache[("out", i)], "ffn:%d" % i
    xi = cache[("in", i)]
    vf = torch.empty_like(xi) if i == 0 else cache["v_first"]
    att, _ = M._tmix_forward(ln(xi, M.ln1_w[i], M.ln1_b[i]), vf, M.tmix[i], i, C16)
    return att, cache[("mid", i)], "att:%d" % i


rows = {}
t2 = time.time()
for arm, sel in by_key.items():
    b, xres, p = local(arm)
    saved = []
    for obj, attr, group, key in sel:
        w = getattr(obj, attr)
        if w is None:
            continue
        qq = fake_quant.q(w, group, cfg, key)
        if qq is not w:
            saved.append((obj, attr, w)); setattr(obj, attr, qq.to(w.dtype))
    if not saved:
        continue
    if arm == "head.weight":
        r = {"score": kl_of(cache["h"] @ M.head_weight.T), "point": "head", "rel2": None, "G": None}
    else:
        a, _, _ = local(arm)
        d = a - b
        rel2 = float((d.norm(dim=-1) ** 2 / xres.norm(dim=-1) ** 2).mean())
        r = {"rel2": rel2, "point": p, "G": g(p), "score": rel2 * g(p)}
    for obj, attr, w in saved:
        setattr(obj, attr, w)
    rows[arm] = r
print("локально: %d плеч за %.0f с" % (len(rows), time.time() - t2), flush=True)
res["rows"] = rows
json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
top = sorted(rows.items(), key=lambda kv: -kv[1]["score"])[:12]
for k, v in top:
    print("  %.6f  %s" % (v["score"], k))
print("ГОТОВО за %.0f с" % (time.time() - t0), flush=True)
