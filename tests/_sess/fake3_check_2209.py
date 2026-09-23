"""Гейт: fake-путь при 3 битах (22.09 ночь, предусловие лестницы вниз).
Реальные формы 0.1B (att.output [768,768], ffn.value [768,3072], ffn.key
[3072,768]), схема COMPRESSION (asym_sb6_aw, gs 32) через q() c bits_overrides.
Проверяет: (1) коды 0..7, деквант восстанавливается из частей; (2) в каждом
блоке <= 8 различных значений; (3) q() с переопределением 3 == прямой вызов;
(4) ошибка по битам монотонна, отношение RMS на бит ~2 (печать, не порог);
(5) 4 бита через override == 4 бита пресета (правка ничего не сдвинула)."""
import copy, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch
from rwkv_quant import presets
from rwkv_quant.calibration import fake_quant, groupwise as gw
CK = os.path.expanduser(os.environ.get("CK", "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
ACT = os.path.expanduser(os.environ.get("ACT", "~/.cache/rwkv-quant/act_stats/act_19fa07321aa994e0.pt"))
sd = torch.load(CK, map_location="cpu", mmap=True, weights_only=True)
cfg = copy.deepcopy(presets.COMPRESSION); cfg.act_stats_path = ACT
fails = 0
for key, grp in [("blocks.5.att.output.weight", "proj"), ("blocks.5.ffn.value.weight", "cmix"),
                 ("blocks.5.ffn.key.weight", "cmix")]:
    w = sd[key].float()
    ex2 = gw.get_ex2(ACT, key, w)
    assert ex2 is not None, "нет статистики для %s" % key
    parts = gw.groupwise_fake_dequant(w, 3, 32, sb=8, sb_bits=-6, ex2=ex2, return_parts=True)
    q, deq = parts["q"], parts["deq"]
    ok1 = int(q.max()) <= 7 and int(q.min()) >= 0
    OUT, IN = w.shape
    rec = (q.float().view(OUT, IN // 32, 32) * parts["scale"] + parts["mn"]).view(OUT, IN)
    ok1 = ok1 and torch.equal(rec, deq)
    nd = max(len(torch.unique(deq.view(OUT, IN // 32, 32)[r, b])) for r in range(0, OUT, 97) for b in range(IN // 32))
    ok2 = nd <= 8
    c3 = copy.copy(cfg); c3.bits_overrides = dict({key: 3}, **cfg.bits_overrides)
    via_q = fake_quant.q(w, grp, c3, key)
    ok3 = torch.equal(via_q, deq)
    ev = ex2.float().clamp_min(1e-12)
    errs = {}
    for b in (3, 4, 5, 6):
        cb = copy.copy(cfg); cb.bits_overrides = dict({key: b}, **cfg.bits_overrides)
        d = fake_quant.q(w, grp, cb, key) - w
        errs[b] = float(((d ** 2) * ev).sum() / ((w ** 2) * ev).sum()) ** 0.5
    ok4 = errs[3] > errs[4] > errs[5] > errs[6]
    b0 = 5 if "ffn.key" in key else 4
    ok5 = torch.equal(fake_quant.q(w, grp, cfg, key), fake_quant.q(w, grp, copy.copy(c3) if False else
                      (lambda c: (setattr(c, "bits_overrides", dict({key: b0}, **cfg.bits_overrides)), c)[1])(copy.copy(cfg)), key))
    r = " ".join("%d:%.4f" % (b, e) for b, e in errs.items())
    print("%-28s коды/деквант %s  <=8 знач. %s (%d)  q()==прямой %s  монотонно %s  база==override %s | отн.ошибка(AW) %s | e3/e4 %.2f e4/e5 %.2f e5/e6 %.2f"
          % (key, ok1, ok2, nd, ok3, ok4, ok5, r, errs[3] / errs[4], errs[4] / errs[5], errs[5] / errs[6]))
    fails += not (ok1 and ok2 and ok3 and ok4 and ok5)
print("ИТОГ:", "ЗЕЛЁНЫЙ" if not fails else "КРАСНЫЙ (%d)" % fails)
sys.exit(1 if fails else 0)
