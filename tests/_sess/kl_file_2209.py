"""KL ФАЙЛА .rwkvq К fp32-ЭТАЛОНУ НА ВСЕХ 38 ОКНАХ (22.09). torch fp32 на ОДНОЙ
карте, веса квантованных точек -- из файла (деквант как в file_weights_arm:
reader.dequantize_banded, LoRA транспонируется); точки bits>=16 остаются из
чекпоинта (bf16 без изменений). Меряется САМ файл, а не fake-путь.
Контроль: KL без подмены на двух окнах обязан быть ~0 (тот же прибор, что эталон),
и подмена обязана сдвинуть веса.
Окна выбора правила (weakspot WIN 0,6,13,20,24,28,30,35) и отложенные 30 --
отдельно; per-window пишется в JSON для ПАРНЫХ разностей между файлами.
    RWKVQ_CKPT=<pth> RWKVQ_KL_REF=<38x512 npy> python kl_file_2209.py <файл> <cuda:k> <out.json>
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, torch
import ablate_sym_composite as comp
from rwkv_quant.calibration.group_config import QuantConfig
from rwkv_quant.formats import reader
from rwkv_quant.models.rwkv7_ref import RWKV7Ref

PATH, DEV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEL = [0, 6, 13, 20, 24, 28, 30, 35]
C16 = QuantConfig()
t0 = time.time()
M = RWKV7Ref(comp.CKPT, device=DEV, dtype=torch.float32)
blob = torch.load(comp.CORPUS)
tok = blob["tokens"][:, :512].contiguous()[:, :-1]
langs = list(blob.get("lang", [])) if isinstance(blob, dict) else []
ref = np.load(os.environ["RWKVQ_KL_REF"], mmap_mode="r")
assert ref.shape[0] == tok.shape[0] and ref.shape[1] == tok.shape[1], (ref.shape, tok.shape)

@torch.no_grad()
def kl_windows(idx, bs=4):
    out = []
    for b0 in range(0, len(idx), bs):
        w = idx[b0:b0 + bs]
        lg = M.forward(tok[w].to(DEV), C16).float()
        for j, wi in enumerate(w):
            lp = torch.log_softmax(torch.from_numpy(np.array(ref[wi])).to(DEV).double(), -1)
            lq = torch.log_softmax(lg[j].double(), -1)
            kl = float((lp.exp() * (lp - lq)).sum(-1).mean())
            top = float((lp.argmax(-1) == lq.argmax(-1)).float().mean())
            out.append((wi, kl, top))
        del lg
    return out

floor = kl_windows([0, 1], bs=2)
print("пол (без подмены): KL %s" % ["%.2e" % k for _, k, _ in floor], flush=True)
assert max(k for _, k, _ in floor) < 1e-5, "прибор разошёлся с эталоном"

ck = reader.load_raw(PATH)
n, moved, skip = 0, 0.0, 0
for obj, attr, group, key in comp.quant_points(M):
    w = getattr(obj, attr)
    if w is None:
        continue
    qt = ck.tensors.get(key)
    if qt is None or qt.bits >= 16:
        skip += 1; continue
    o = reader.dequantize_banded(qt, torch.float32)
    if tuple(o.shape) == tuple(w.shape)[::-1] != tuple(w.shape):
        o = o.T.contiguous()
    assert tuple(o.shape) == tuple(w.shape), (key, o.shape, w.shape)
    o = o.to(w.device)
    moved += float((o - w.float()).pow(2).sum()); setattr(obj, attr, o.to(w.dtype)); n += 1
    del w, o
torch.cuda.empty_cache()
assert n > 0 and moved > 0, "подмена ничего не изменила"
print("подменено %d, из чекпоинта %d, энергия сдвига %.4e, %.0f с" % (n, skip, moved, time.time() - t0), flush=True)

res = kl_windows(list(range(tok.shape[0])))
kl = np.array([k for _, k, _ in res]); top = np.array([t for _, _, t in res])
ho = np.array([i not in SEL for i in range(len(kl))])
def ci(x):
    r = np.random.default_rng(0); b = x[r.integers(0, len(x), (20000, len(x)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
lo, hi = ci(kl)
print("%s: KL %.6f [%.6f; %.6f]  top-1 %.3f%%  | выбор(8) %.6f  отложенные(30) %.6f" % (
    os.path.basename(PATH), kl.mean(), lo, hi, 100 * top.mean(), kl[~ho].mean(), kl[ho].mean()), flush=True)
if langs:
    for L_ in sorted(set(langs)):
        m = np.array([l == L_ for l in langs])
        print("   %s (%d окон): %.6f" % (L_, m.sum(), kl[m].mean()))
json.dump(dict(file=PATH, size=os.path.getsize(PATH), kl=kl.tolist(), top1=top.tolist(), langs=langs,
               sel=SEL, floor=[k for _, k, _ in floor]), open(OUT, "w"))
print("ГОТОВО за %.0f с" % (time.time() - t0), flush=True)
