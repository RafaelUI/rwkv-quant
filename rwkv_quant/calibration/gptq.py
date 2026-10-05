"""GPTQ на сетке asym_sb6 (28.09) и на симметричной сетке sym (01.10): компенсация ошибки округления для
COMPRESSION и REDUCTION без смены формата.

sym (REDUCTION, решение владельца 01.10): Q6_K-подобная раскладка writer'а (_sym_one: scale на блок gs без min,
int8-коды scale против fp16 d на суперблок из 256 // gs блоков, 6/8 бит). Сетка суперблока строится _sym_one по
уже поправленным весам в его начале, дальше -- тот же цикл. Упаковка -- writer._pack_gw_sym. Цифры
(прототип tests/_sess/gptq_srv_3009, 600 окон, damp 0.1, бутстрэп по окнам): KL к RTN текст / код 0.1B -22 / -28,
0.4B -23 / -27, 1.5B -28 / -32, 2.9B -30 / -34, 7.2B -21 / -24%, значимо на каждом языке; ppl -- в шуме.


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


def gptq_parts_sym(W, H, bits, ex2=None, damp=0.1, gs=16, sb=16, search=True):
    """sym-сетка (как writer._make_qt_gw_sym) + компенсация ошибки. W [OUT, IN] fp32 на устройстве счёта,
    H [IN, IN] (CPU) -> {q int8 [OUT, IN], qs int8 [OUT, NB], d fp16 [OUT, NSB]} на CPU + deq [OUT, IN] fp32 на
    устройстве W. Проверка 0: при диагональной H части == _sym_one побитно (гейт test_quantize_gptq)."""
    OUTd, IN = W.shape
    BL = gs * sb
    assert IN % BL == 0, IN
    W = W.clone()
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead.to(W.device)] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(IN, dtype=H.dtype)
    # нижний фактор + .mT -- не cholesky(upper=True) (порча кучи на macOS, см. gptq_parts)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H))).mT.contiguous().float().to(W.device)
    Q = torch.zeros_like(W)
    codes = torch.zeros(OUTd, IN, dtype=torch.int8, device=W.device)
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    qs_l, d_l = [], []
    for b0 in range(0, IN, BL):
        b1 = b0 + BL
        W1 = W[:, b0:b1].clone()
        E1 = torch.zeros_like(W1)
        C1 = torch.zeros_like(W1)
        Hi = Hinv[b0:b1, b0:b1]
        p = gw._sym_one(W1, bits, gs, sb, ex2[b0:b1] if ex2 is not None else None, search, 0.0, True)
        qs_l.append(p["qs"]); d_l.append(p["d"])
        # == scale_q в _sym_one: (qs * d).half(); у вырожденного суперблока qs = 0 (nan_to_num) -> scale 0 -> nz False
        sc = (p["qs"].float() * p["d"].float().repeat_interleave(sb, dim=1)).half().float()
        nz = sc.abs() > 0
        den = torch.where(nz, sc, torch.ones_like(sc))
        for j in range(BL):
            g = j // gs
            w = W1[:, j]
            c = torch.clamp(torch.round(w / den[:, g]), qmin, qmax)
            c = torch.where(nz[:, g], c, torch.zeros_like(c))
            qv = c * sc[:, g]
            Q[:, b0 + j] = qv
            C1[:, j] = c
            e = (w - qv) / Hi[j, j]
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        codes[:, b0:b1] = C1.to(torch.int8)
        W[:, b1:] -= E1 @ Hinv[b0:b1, b1:]
    cat = lambda xs: torch.cat([x.cpu() for x in xs], dim=1).contiguous()
    return dict(q=codes.cpu(), qs=cat(qs_l), d=cat(d_l), deq=Q)


N_WINDOWS = 600      # решение владельца 30.09: 600 окон всегда (помогает и малым: +3-4 п. на 1.5B/2.9B)
DAMP = 0.1           # 29.09: единое значение; 0.01 переобучает (7.2B/13.3B хуже RTN на en) и хуже на малых
CALIB_FILE = os.path.join(A.DATA_DIR, "gptq_calib.jsonl")


def devices(device=None):
    """"cuda:0,cuda:1" / список / одно устройство / None -> список устройств. None: RWKVQ_DEVICE, иначе cuda > mps > cpu.
    Слои разносятся по списку как в RWKV7Ref (слой i -> devices[i * k // n_layer])."""
    if device is None:
        device = os.environ.get("RWKVQ_DEVICE") or ("cuda" if torch.cuda.is_available() else
                                                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    if isinstance(device, str):
        device = [d.strip() for d in device.split(",") if d.strip()]
    return [str(d) for d in device]


def calib_tokens(tokenizer, calib=None, n_windows=N_WINDOWS, seq_len=A.SEQ_LEN):
    """-> (LongTensor [n, seq_len] на CPU, описание источника).
    calib: None -- корпус пакета (CALIB_FILE: по окну текста на строку JSONL, разбирается словарём МОДЕЛИ -- свой
    токенизатор библиотека не везёт); путь к .pt ({"tokens": [N, T]} или тензор [N, T]); тензор или список списков.
    Берутся ПЕРВЫЕ n_windows окон: порядок корпуса -- взвешенный круг по группам, любой префикс держит доли."""
    if calib is None:
        if not os.path.exists(CALIB_FILE):
            raise FileNotFoundError("корпус GPTQ пакета не найден: %s" % CALIB_FILE)
        import json
        enc = A._encoder(tokenizer)
        rows, short = [], 0
        for line in open(CALIB_FILE, encoding="utf-8"):
            ids = enc(json.loads(line)["text"])
            if len(ids) < seq_len:
                short += 1
                continue
            rows.append(ids[:seq_len])
            if len(rows) == n_windows:
                break
        label = "package:%s" % os.path.basename(CALIB_FILE)
        if short:
            label += " (коротких окон пропущено: %d)" % short
        tok = torch.tensor(rows, dtype=torch.long)
    else:
        label = "tensor"
        if isinstance(calib, str):
            label = "file:%s" % os.path.basename(calib)
            calib = torch.load(os.path.expanduser(calib), map_location="cpu", weights_only=False)
            calib = calib["tokens"] if isinstance(calib, dict) else calib
        tok = torch.as_tensor(calib, dtype=torch.long)[:n_windows, :seq_len].cpu().contiguous()
    if tok.dim() != 2 or tok.shape[0] < n_windows or tok.shape[1] < 2:
        raise ValueError(
            "калибровка GPTQ: нужен тензор токенов [N, T] с N >= %d окон и T >= 2, передано %s. "
            "Берутся первые %d окон и первые %d токенов каждого; число окон фиксировано, меньший "
            "набор не принимается (без gptq_calib используется корпус пакета)."
            % (n_windows, tuple(tok.shape), n_windows, seq_len))
    return tok, label


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)
    elif str(dev).startswith("mps"):
        # без синхронизации MPS копит сотни тысяч мелких запусков _wkv7 в одном
        # командном буфере и падает на выделении IOGPUDeviceShmem (27.09)
        torch.mps.synchronize()


def run(ckpt, tokenizer, cfg, n_windows=N_WINDOWS, damp=DAMP, device=None, verbose=True, keep_deq=False, calib=None):
    """GPTQ всех матриц proj/cmix/head, которые cfg пишет в sb6 (asym_sb6*, 4-6 бит; bits_overrides
    учитываются -- как writer: подстрока, первое совпадение). -> {key: QuantizedTensor}, упакованные
    writer._pack_gw_sb6. keep_deq=True дополнительно возвращает {key: deq fp32 CPU} (для гейта).
    cfg.act_stats_path обязателен для asym_sb6_aw (ex2 -- та же статистика, что у writer).

    30.09 (под API): device -- строка "cuda:0,cuda:1" или список (слои по картам); калибровка -- calib_tokens;
    активации калибровки ВСЕГДА на CPU и идут на устройство слоя батчами по 8 (сотни окон на 7.2B/13.3B не
    влезают в карту); итог прохода пишется на место (пик -- два тензора [N, T, C] fp32: x и v_first).
    Цикл GPTQ -- на карте слоя, если это CUDA, иначе на CPU (как writer). H головы копится батчами."""
    from ..formats import writer as Wr
    from ..models import rwkv7_ref as R

    t0 = time.time()
    devs = devices(device)
    stats = gw.load_act_stats(cfg.act_stats_path) if getattr(cfg, "act_stats_path", None) else {}
    M = R.RWKV7Ref(ckpt, device=(devs if len(devs) > 1 else devs[0]), dtype=torch.bfloat16, compute_dtype=torch.float32)
    if isinstance(calib, torch.Tensor) and calib.shape[0] == n_windows and calib.dim() == 2:
        cal = calib.long().cpu()
    else:
        cal, _ = calib_tokens(tokenizer, calib, n_windows)
    cal = cal[:, :-1].contiguous()
    N, T, C = cal.shape[0], cal.shape[1], M.n_embd
    BS = 8
    out, deqs, done, Hs = {}, {}, {}, {}
    loop_dev = lambda d: d if str(d).startswith("cuda") else "cpu"

    def plan(group, key):
        """("sb6", bits, gs, sb_bits, ex2) / ("sym", bits, gs, search, ex2) или None, если матрица не идёт ни в
        sb6, ни в sym (как решает writer.quantize_tensor)."""
        bits = ap._bits_of(cfg, group, key)
        gs = cfg.group_scale.get(group)
        mode = cfg.group_scale_mode.get(group, "asym")
        if not gs or bits >= 16:
            return None
        # ex2 -- ровно как writer.quantize_tensor: sb6 -- только "asym_sb6_aw"; sym -- mode.endswith("_aw")
        if mode in _SB_BITS and bits in (4, 5, 6):
            kind, aw, x = "sb6", mode == "asym_sb6_aw", _SB_BITS[mode]
        elif mode.startswith("sym") and bits in (6, 8) and not cfg.outlier_fracs.get(group, 0.0):
            kind, aw, x = "sym", mode.endswith("_aw"), not mode.endswith("_plain")
        else:
            return None
        ex2 = stats.get(key) if aw else None
        if aw and ex2 is None:
            return None               # writer без статистики уйдёт в поиск без AW -- не подменяем
        return kind, bits, gs, x, ex2

    def solve(W, H, pl, key, group):
        """-> (QuantizedTensor, deq) по плану."""
        kind, bits, gs, x, ex2 = pl
        OUTd, IN = W.shape
        if kind == "sb6":
            p = gptq_parts(W, H, bits, ex2, damp, gs, x)
            Q = p.pop("deq")
            return Wr._pack_gw_sb6(key, group, bits, OUTd, IN, gs, p), Q
        sb = max(1, 256 // gs)
        p = gptq_parts_sym(W, H, bits, ex2, damp, gs, sb, x)
        Q = p.pop("deq")
        return Wr._pack_gw_sym(key, group, bits, OUTd, IN, gs, sb, p), Q

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

    def do(obj, attr, key, group, H, dev):
        pl = plan(group, key)
        if pl is None:
            return
        W0 = getattr(obj, attr)
        W = W0.float().to(loop_dev(dev))
        out[key], Q = solve(W, H, pl, key, group)
        if keep_deq:
            deqs[key] = Q.cpu()
        setattr(obj, attr, Q.to(W0.device))   # fp32: деквант без bf16-округления, как у RTN-пути
        done[key] = 16
        # то, что видит forward, обязано быть ровно Q (28.09: повторное квантование при autopick)
        if not torch.equal(M._q(getattr(obj, attr), group, cseq(), key), getattr(obj, attr)):
            raise AssertionError("gptq: forward квантует GPTQ-веса повторно: " + key)

    def att_pass(i, write):
        dv = M.layer_dev[i]
        for b in range(0, N, BS):
            xb, vb = x[b:b + BS].to(dv), vf[b:b + BS].to(dv)
            xn = F.layer_norm(xb, (C,), M.ln1_w[i].float(), M.ln1_b[i].float())
            a, v2 = M._tmix_forward(xn, vb, M.tmix[i], i, cseq())
            _sync(dv)
            if write:
                y = xb + a
                x[b:b + BS] = y.cpu()
                vf[b:b + BS] = v2.cpu()

    def ffn_pass(i, write):
        dv = M.layer_dev[i]
        for b in range(0, N, BS):
            xb = x[b:b + BS].to(dv)
            xn = F.layer_norm(xb, (C,), M.ln2_w[i].float(), M.ln2_b[i].float())
            y = xb + M._cmix_forward(xn, M.cmix[i], cseq(), i)
            _sync(dv)
            if write:
                x[b:b + BS] = y.cpu()

    prev_rec = R._rec
    R._rec = rec
    try:
        with torch.no_grad():
            x = torch.empty(N, T, C)
            vf = torch.zeros(N, T, C)
            d0 = M.emb_weight.device
            E = M._q(M.emb_weight, "emb", cfg, "emb.weight")
            for b in range(0, N, BS):
                e = F.embedding(cal[b:b + BS].to(d0), E)
                x[b:b + BS] = F.layer_norm(e.float(), (C,), M.ln0_w.float(), M.ln0_b.float()).cpu()
            del E
            for i in range(M.n_layer):
                t, c = M.tmix[i], M.cmix[i]
                p = "blocks.%d." % i
                dv = M.layer_dev[i]
                H = collect({p + "att.receptance.weight": t.r_proj.shape[1], p + "att.key.weight": t.k_proj.shape[1],
                             p + "att.value.weight": t.v_proj.shape[1]}, lambda: att_pass(i, False))
                for attr, nm in (("r_proj", "receptance"), ("k_proj", "key"), ("v_proj", "value")):
                    do(t, attr, p + "att.%s.weight" % nm, "proj", H[p + "att.%s.weight" % nm], dv)
                H = collect({p + "att.output.weight": t.o_proj.shape[1]}, lambda: att_pass(i, False))
                do(t, "o_proj", p + "att.output.weight", "proj", H[p + "att.output.weight"], dv)
                att_pass(i, True)
                H = collect({p + "ffn.key.weight": c.key.shape[1]}, lambda: ffn_pass(i, False))
                do(c, "key", p + "ffn.key.weight", "cmix", H[p + "ffn.key.weight"], dv)
                H = collect({p + "ffn.value.weight": c.value.shape[1]}, lambda: ffn_pass(i, False))
                do(c, "value", p + "ffn.value.weight", "cmix", H[p + "ffn.value.weight"], dv)
                ffn_pass(i, True)
                # слой пройден: его веса больше не нужны -- fp32-копия модели не копится
                if FREE:
                    for obj, attr in ((t, "r_proj"), (t, "k_proj"), (t, "v_proj"), (t, "o_proj"), (c, "key"), (c, "value")):
                        setattr(obj, attr, torch.empty(0, device=getattr(obj, attr).device))
                if not torch.isfinite(x).all():
                    raise FloatingPointError("gptq: нечисло в активациях после слоя %d" % i)
                if verbose:
                    print("[gptq] слой %d/%d, %.0f с" % (i + 1, M.n_layer, time.time() - t0), flush=True)
            dl = M.ln_out_w.device
            Hh = torch.zeros(C, C)
            for b in range(0, N, BS):
                xo = F.layer_norm(x[b:b + BS].to(dl), (C,), M.ln_out_w.float(), M.ln_out_b.float()).reshape(-1, C)
                Hh += (xo.T @ xo).cpu()
            del x, vf
            pl = plan("head", "head.weight")
            if pl is not None:
                W = M.head_weight.float().to(loop_dev(dl))
                out["head.weight"], Q = solve(W, Hh, pl, "head.weight", "head")
                if keep_deq:
                    deqs["head.weight"] = Q.cpu()
    finally:
        R._rec = prev_rec
    del M
    if verbose:
        print("[gptq] %d матриц, %d окон x %d, damp %g, %.0f с" % (len(out), N, T, damp, time.time() - t0), flush=True)
    return (out, deqs) if keep_deq else out
