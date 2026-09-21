"""ЭТАЛОН ДЛЯ АВТОПОДБОРА (21.09): вред КАЖДОЙ матрицы по отдельности.

Квантуется РОВНО одна матрица (или LoRA одного слоя целиком), всё прочее --
исходные веса; KL против fp32-эталона на окнах, РАЗНЕСЁННЫХ ПО ЯЗЫКАМ
(первые 8 окон корпуса -- целиком русские, раздел 19.09). Модель одна на
процесс, веса подменяются и возвращаются, поэтому одна загрузка на шард.

Активации в fp32 по умолчанию (закон 39): bf16-плечо даёт собственный
пол KL того же порядка, что вред мелких матриц (--only cmix_v_l15 на 2.9B
в bf16 дал 0.000776 -- при неизвестном поле это число не читается).
Пол меряется плечом __none__ (ничего не квантовано) в каждом шарде 0.

    RWKVQ_CKPT=... RWKVQ_ACT_STATS=... RWKVQ_KL_REF=... RWKVQ_CORPUS=... \
    RWKVQ_DEVICE=cuda:0 python only_sweep_2109.py <конфиг> <шард>/<из> <выход.json>
Окна: RWKVQ_SWEEP_WIN (по умолчанию 0,6,13,20,24,28,30,35 = 3 ru, 3 en, 2 sr).
Тип: RWKVQ_SWEEP_DTYPE fp32|bf16.
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np
import torch
import ablate_sym_composite as comp
from rwkv_quant.calibration import fake_quant
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

name, shard, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
si, sn = (int(x) for x in shard.split("/"))
DEV = os.environ.get("RWKVQ_DEVICE", "cuda:0")
DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("RWKVQ_SWEEP_DTYPE", "fp32")]
WIN = [int(x) for x in os.environ.get("RWKVQ_SWEEP_WIN", "0,6,13,20,24,28,30,35").split(",")]
QGROUPS = ("proj", "cmix", "emb", "head")          # по одной матрице
LORA = ("w_lora", "a_lora", "v_lora", "g_lora")    # LoRA слоя -- одним плечом

cfg = comp.CONFIGS[name]()
if not os.path.exists(comp.ACT):
    raise SystemExit("нет статистики %s (закон 15)" % comp.ACT)
blob = torch.load(comp.CORPUS)
langs = [blob["lang"][w] for w in WIN]
data = blob["tokens"][WIN, :512].contiguous().to(DEV)
ref = np.load(os.environ["RWKVQ_KL_REF"], mmap_mode="r")
assert ref.shape[0] >= max(WIN) + 1 and ref.shape[1] == 511, ref.shape
REF = [torch.from_numpy(np.asarray(ref[w])).to(DEV) for w in WIN]      # fp32, ~134 МБ/окно

t0 = time.time()
model = RWKV7Ref(comp.CKPT, device=DEV, dtype=DT)
pts = comp.quant_points(model)
print("[%s] загрузка %.0f с, окна %s (%s), тип %s" % (DEV, time.time() - t0, WIN, "".join(l[0] for l in langs), DT), flush=True)

arms = []
for obj, attr, group, key in pts:
    if group in QGROUPS:
        arms.append((key, [(obj, attr, group, key)]))
by_layer = {}
for obj, attr, group, key in pts:
    if group in LORA:
        by_layer.setdefault(int(key.split(".")[1]), []).append((obj, attr, group, key))
arms += [("blocks.%d.lora" % i, v) for i, v in sorted(by_layer.items())]
arms = [a for j, a in enumerate(arms) if j % sn == si]
if si == 0:
    arms = [("__none__", [])] + arms + [("__all__", [p for p in pts])]


@torch.no_grad()
def measure():
    kls, tops = [], []
    for i in range(len(WIN)):
        lg = model.forward(data[i:i + 1, :-1])[0].double()
        P = REF[i].double()
        lp = torch.log_softmax(P, -1); lq = torch.log_softmax(lg, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        if not torch.isfinite(kl).all():
            raise RuntimeError("не-конечный KL, окно %d" % WIN[i])
        kls.append(float(kl.mean())); tops.append(float((P.argmax(-1) == lg.argmax(-1)).double().mean()))
        del lg, P, lp, lq, kl
    return kls, tops


res = json.load(open(out_path)) if os.path.exists(out_path) else {}
res.setdefault("_meta", dict(ckpt=comp.CKPT, cfg=name, win=WIN, langs=langs, dtype=str(DT), act=comp.ACT))
for arm, sel in arms:
    if arm in res:
        continue
    t1 = time.time()
    saved = []
    for obj, attr, group, key in sel:
        w = getattr(obj, attr)
        if w is None:
            continue
        q = fake_quant.q(w, group, cfg, key)
        if q is not w:
            saved.append((obj, attr, w)); setattr(obj, attr, q.to(w.dtype))
    if sel and not saved:
        res[arm] = {"skip": "не квантуется в %s" % name}
        continue
    kls, tops = measure()
    for obj, attr, w in saved:
        setattr(obj, attr, w)
    res[arm] = {"kl": float(np.mean(kls)), "per_win": kls, "top1": float(np.mean(tops)),
                "n": len(saved), "s": round(time.time() - t1, 1)}
    print("%-34s KL %.6f  top1 %.4f  %4.1f с" % (arm, res[arm]["kl"], res[arm]["top1"], time.time() - t1), flush=True)
    json.dump(res, open(out_path + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(out_path + ".tmp", out_path)
print("ГОТОВО шард %s за %.0f с" % (shard, time.time() - t0), flush=True)
