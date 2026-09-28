"""GPTQ на сетке asym_sb6 (28.09): компенсация ошибки округления для COMPRESSION без смены формата.

Сетка -- ТА ЖЕ, что у writer (_gw_one: блок gs, суперблок 8, 6-битные scale/min, поиск scale; для
asym_sb6_aw -- с весами E[x^2]). scale/min суперблока (8*gs колонок) считаются В НАЧАЛЕ суперблока по уже
поправленным весам, затем колонки квантуются по одной на эту сетку, а ошибка каждой колонки разносится
на ещё не квантованные через H^-1 (Frantar et al. 2022). В файл идут те же коды/qs/qm/d/dm, упакованные
writer._pack_gw_sb6 -- формат и кернели не меняются.

Проход по слоям -- как в прототипе tests/_sess/gptq_proto_2709.py (цифры 27-28.09): вход каждой матрицы
снимается с модели, где предыдущие матрицы уже квантованы (r,k,v -> o -> ffn.key -> ffn.value; голова --
в конце, emb -- обычным путём writer). Отличие от прототипа -- память: веса пройденного слоя
освобождаются, fp32-копия модели не копится (пик -- bf16-модель + один слой fp32 + H одной матрицы).

Проверка 0 (гейт test_quantize_gptq): при диагональной H части GPTQ ПОБИТНО равны частям _gw_one."""
import copy
import os
import re
import time

import torch
import torch.nn.functional as F

from . import act_stats as A
from . import autopick as ap
from . import groupwise as gw

SB = 8
FREE = os.environ.get("RWKVQ_GPTQ_FREE", "1") == "1"
_SB_BITS = {"asym_sb6_aw": -6, "asym_sb6_search": -6, "asym_sb6": 6}


def gptq_parts(W, H, bits, ex2=None, damp=0.01, gs=32, sb_bits=-6):
    """W [OUT, IN] fp32 (на устройстве счёта), H [IN, IN] fp32/fp64 (CPU) ->
    части sb6 {q uint8 [OUT, IN], qs, qm [OUT, NB], d, dm [OUT, NSB]} на CPU + deq [OUT, IN] fp32
    (на устройстве W) -- ровно то, что даст деквант записанного файла (до bf16)."""
    OUTd, IN = W.shape
    BL = gs * SB
    assert IN % BL == 0, IN
    W = W.clone()
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead.to(W.device)] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(IN, dtype=H.dtype)
    # 28.09: НЕ torch.linalg.cholesky(..., upper=True) -- на torch 2.13 / macOS (Accelerate) он портит кучу:
    # пишет за свой буфер (порча уже упакованных кодов, segfault в MPSGraph, "objc hash table corrupted",
    # зависание в поиске кеша графов, segfault на выходе процесса). Нижний фактор + транспонирование --
    # U побитно тот же (12 из 12, 256..3072); на CUDA/MKL -- до 1e-15. Гейт: test_quantize_gptq, часть 2 (коды файла == deq после всего прохода).
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H))).mT.contiguous().float().to(W.device)
    Q = torch.zeros_like(W)
    codes = torch.zeros(OUTd, IN, dtype=torch.uint8, device=W.device)
    qmax = 2 ** bits - 1
    qs_l, qm_l, d_l, dm_l = [], [], [], []
    for b0 in range(0, IN, BL):
        b1 = b0 + BL
        W1 = W[:, b0:b1].clone()
        E1 = torch.zeros_like(W1)
        C1 = torch.zeros_like(W1)     # коды суперблока (float): см. ниже
        Hi = Hinv[b0:b1, b0:b1]
        p = gw._gw_one(W1, bits, gs, SB, sb_bits, ex2[b0:b1] if ex2 is not None else None, True)
        sc = p["scale"].view(OUTd, -1)
        mn = p["mn"].view(OUTd, -1)
        qs_l.append(p["qs"].view(OUTd, -1)); qm_l.append(p["qm"].view(OUTd, -1))
        d_l.append(p["d"].view(OUTd, -1)); dm_l.append(p["dm"].view(OUTd, -1))
        for j in range(BL):
            g = j // gs
            w = W1[:, j]
            c = torch.clamp(torch.round((w - mn[:, g]) / sc[:, g]), 0, qmax)
            qv = c * sc[:, g] + mn[:, g]
            Q[:, b0 + j] = qv
            C1[:, j] = c
            e = (w - qv) / Hi[j, j]
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        # коды копятся во float (C1) и переводятся в uint8 одной операцией на суперблок
        codes[:, b0:b1] = C1.to(torch.uint8)
        W[:, b1:] -= E1 @ Hinv[b0:b1, b1:]
    cat = lambda xs: torch.cat([x.cpu() for x in xs], dim=1).contiguous()
    return dict(q=codes.cpu(), qs=cat(qs_l), qm=cat(qm_l), d=cat(d_l), dm=cat(dm_l), deq=Q)


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)
    elif str(dev).startswith("mps"):
        # без синхронизации MPS копит сотни тысяч мелких запусков _wkv7 в одном
        # командном буфере и падает на выделении IOGPUDeviceShmem (27.09)
        torch.mps.synchronize()


def run(ckpt, tokenizer, cfg, n_windows=48, damp=0.01, device=None, verbose=True, keep_deq=False):
    """GPTQ всех матриц proj/cmix/head, которые cfg пишет в sb6 (asym_sb6*, 4-6 бит; bits_overrides
    учитываются -- как writer: подстрока, первое совпадение). -> {key: QuantizedTensor}, упакованные
    writer._pack_gw_sb6. keep_deq=True дополнительно возвращает {key: deq fp32 CPU} (для гейта).
    cfg.act_stats_path обязателен для asym_sb6_aw (ex2 -- та же статистика, что у writer)."""
    from ..formats import writer as Wr
    from ..models import rwkv7_ref as R

    t0 = time.time()
    dev = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    stats = gw.load_act_stats(cfg.act_stats_path) if getattr(cfg, "act_stats_path", None) else {}
    M = R.RWKV7Ref(ckpt, device=dev, dtype=torch.bfloat16, compute_dtype=torch.float32)
    enc = A._encoder(tokenizer)
    chunks = [c.strip() for c in re.split(r"—+ CHUNK —+", open(A.CORPUS, encoding="utf-8").read()) if c.strip()]
    cal = A._windows(chunks, enc, A.SEQ_LEN, n_windows * A.SEQ_LEN)[:n_windows]
    cal = torch.tensor(cal, dtype=torch.long)[:, :-1].contiguous().to(dev)
    BS = 8
    out, deqs, done, Hs = {}, {}, {}, {}

    def plan(group, key):
        """(bits, gs, sb_bits, ex2) или None, если матрица не идёт в sb6."""
        bits = ap._bits_of(cfg, group, key)
        gs = cfg.group_scale.get(group)
        mode = cfg.group_scale_mode.get(group, "asym")
        if not gs or bits >= 16 or mode not in _SB_BITS or bits not in (4, 5, 6):
            return None
        ex2 = stats.get(key) if mode == "asym_sb6_aw" else None
        if mode == "asym_sb6_aw" and ex2 is None:
            return None               # writer без статистики уйдёт в поиск без AW -- не подменяем
        return bits, gs, _SB_BITS[mode], ex2

    def cseq():
        # отметки done=16 ПЕРВЫМИ и не перезаписываются выбором (28.09: dict(done, **overrides)
        # отдавал совпавшие ключи выбору autopick -> повторное квантование GPTQ-весов)
        c = copy.copy(cfg)
        c.bits_overrides = dict(done, **{k: v for k, v in cfg.bits_overrides.items() if k not in done})
        return c

    def rec(name, x):
        if name in Hs:
            v = x.detach().float().reshape(-1, x.shape[-1])
            Hs[name] += (v.T @ v).cpu()

    def collect(dims, fn):
        """dims: {ключ: IN} -- размер H по форме весов, не по статистике."""
        for k, n in dims.items():
            Hs[k] = torch.zeros(n, n)
        fn()
        return {k: Hs.pop(k) for k in dims}

    def do(obj, attr, key, group, H):
        pl = plan(group, key)
        if pl is None:
            return
        bits, gs, sbb, ex2 = pl
        W0 = getattr(obj, attr)
        # цикл GPTQ -- на CPU, как и writer (_gw_one по CPU-тензорам): части считаются там же, где их
        # считал бы обычный путь записи. Forward модели остаётся на устройстве. (Падения MPS 28.09,
        # которые сначала списывались на кеш графов, -- следы порчи кучи от cholesky(upper=True), см. ниже.)
        W = W0.float().cpu()
        OUTd, IN = W.shape
        p = gptq_parts(W, H, bits, ex2, damp, gs, sbb)
        Q = p.pop("deq")
        out[key] = Wr._pack_gw_sb6(key, group, bits, OUTd, IN, gs, p)
        if keep_deq:
            deqs[key] = Q
        setattr(obj, attr, Q.to(W0.device))   # fp32: деквант без bf16-округления, как у RTN-пути
        done[key] = 16

    def att_pass(i, x, vf):
        outs, vfs = [], []
        for b in range(0, x.shape[0], BS):
            xb, vb = x[b:b + BS], vf[b:b + BS]
            xn = F.layer_norm(xb, (M.n_embd,), M.ln1_w[i].float(), M.ln1_b[i].float())
            a, v2 = M._tmix_forward(xn, vb, M.tmix[i], i, cseq())
            _sync(dev)
            outs.append(xb + a); vfs.append(v2)
        return torch.cat(outs), torch.cat(vfs)

    def ffn_pass(i, x):
        outs = []
        for b in range(0, x.shape[0], BS):
            xb = x[b:b + BS]
            xn = F.layer_norm(xb, (M.n_embd,), M.ln2_w[i].float(), M.ln2_b[i].float())
            outs.append(xb + M._cmix_forward(xn, M.cmix[i], cseq(), i))
            _sync(dev)
        return torch.cat(outs)

    prev_rec = R._rec
    R._rec = rec
    try:
        with torch.no_grad():
            x = F.embedding(cal, M._q(M.emb_weight, "emb", cfg, "emb.weight"))
            x = F.layer_norm(x.float(), (M.n_embd,), M.ln0_w.float(), M.ln0_b.float())
            vf = torch.empty_like(x)
            for i in range(M.n_layer):
                t, c = M.tmix[i], M.cmix[i]
                p = "blocks.%d." % i
                H = collect({p + "att.receptance.weight": t.r_proj.shape[1], p + "att.key.weight": t.k_proj.shape[1],
                             p + "att.value.weight": t.v_proj.shape[1]}, lambda: att_pass(i, x, vf))
                for attr, nm in (("r_proj", "receptance"), ("k_proj", "key"), ("v_proj", "value")):
                    do(t, attr, p + "att.%s.weight" % nm, "proj", H[p + "att.%s.weight" % nm])
                H = collect({p + "att.output.weight": t.o_proj.shape[1]}, lambda: att_pass(i, x, vf))
                do(t, "o_proj", p + "att.output.weight", "proj", H[p + "att.output.weight"])
                x, vf = att_pass(i, x, vf)
                H = collect({p + "ffn.key.weight": c.key.shape[1]}, lambda: ffn_pass(i, x))
                do(c, "key", p + "ffn.key.weight", "cmix", H[p + "ffn.key.weight"])
                H = collect({p + "ffn.value.weight": c.value.shape[1]}, lambda: ffn_pass(i, x))
                do(c, "value", p + "ffn.value.weight", "cmix", H[p + "ffn.value.weight"])
                x = ffn_pass(i, x)
                # слой пройден: его веса больше не нужны -- fp32-копия модели не копится
                if FREE:
                    for obj, attr in ((t, "r_proj"), (t, "k_proj"), (t, "v_proj"), (t, "o_proj"), (c, "key"), (c, "value")):
                        setattr(obj, attr, torch.empty(0, device=getattr(obj, attr).device))
                if not torch.isfinite(x).all():
                    raise FloatingPointError("gptq: нечисло в активациях после слоя %d" % i)
                if verbose:
                    print("[gptq] слой %d/%d, %.0f с" % (i + 1, M.n_layer, time.time() - t0), flush=True)
            x = x.to(M.ln_out_w.device)
            xo = F.layer_norm(x, (M.n_embd,), M.ln_out_w.float(), M.ln_out_b.float()).reshape(-1, M.n_embd)
            Hh = (xo.T @ xo).cpu()
            del x, vf, xo
            pl = plan("head", "head.weight")
            if pl is not None:
                bits, gs, sbb, ex2 = pl
                W = M.head_weight.float().cpu()
                p = gptq_parts(W, Hh, bits, ex2, damp, gs, sbb)
                Q = p.pop("deq")
                out["head.weight"] = Wr._pack_gw_sb6("head.weight", "head", bits, W.shape[0], W.shape[1], gs, p)
                if keep_deq:
                    deqs["head.weight"] = Q
    finally:
        R._rec = prev_rec
    del M
    if verbose:
        print("[gptq] %d матриц, %.0f с" % (len(out), time.time() - t0), flush=True)
    return (out, deqs) if keep_deq else out
