"""
Спекулятивный декод с ДРАФТ-МОДЕЛЬЮ (g1d-0.1b -> g1h-1.5b): согласие,
скорость, память.

Отличие от spec_decode_champ.py -- там черновик берётся n-gram-поиском по
истории (даром, но работает только на повторяющемся тексте). Здесь
черновик генерирует настоящая маленькая модель.

ПОЧЕМУ ЧЕРНОВИК ТОЖЕ КВАНТУЕТСЯ. Декод упирается в полосу памяти, значит
считать надо байты за токен, а не параметры. Цель (1.5B COMPRESSION) --
878 МБ за токен (970 МБ файла минус emb: она gather одной строки, а не
проход). Черновик 0.1B в bf16 -- 380 МБ, минус emb ~280 МБ. Отношение
3.1x: при K=4 черновик потратит 1120 МБ против 878 МБ у одной верификации
цели, то есть съест больше, чем сэкономит. Тот же черновик под
COMPRESSION -- ~76 МБ за токен, отношение 11.5x, и арифметика сходится.
Ценой падения качества черновика, что и меряется здесь как согласие.

СОСТОЯНИЕ. RWKV хранит state, а не KV-кэш, поэтому откатить его на
середину раунда нельзя. Механика pending из spec_decode_champ.py:
state берётся новый ТОЛЬКО при полном принятии черновика, иначе остаётся
старый, а принятые токены уезжают в pending и догоняют state следующим
проходом. Черновик ведёт своё состояние тем же способом.

ПАМЯТЬ. Модели строятся по очереди, torch-состояние освобождается до
таймингов, своп замеряется до и после -- если он вырос, замер скорости
недействителен и об этом печатается предупреждение.

    python tests/spec_decode_draft_model.py [K] [N_GEN]
"""
import copy
import gc
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mlx.core as mx  # noqa: E402

from rwkv_quant.presets import COMPRESSION  # noqa: E402
from rwkv_quant.formats.writer import quantize_tensor  # noqa: E402
from rwkv_quant.formats.schema import QuantizedCheckpoint  # noqa: E402
from rwkv_quant.backends.metal.quant_model import QuantRWKV7  # noqa: E402

TARGET_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
DRAFT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")
ACT_STATS = "/tmp/act_stats_1p5b_multiling.pt"

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 192
FLUSH = 2 * K


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                       capture_output=True, text=True).stdout
    return float(s.split("used =")[1].split("M")[0])


def mem(tag):
    mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                        text=True).stdout
    free = [l for l in mp.splitlines() if "free percentage" in l]
    print(f"  [mem/{tag}] swap used {swap_mb():.0f}M | "
          f"{free[0].strip() if free else ''}", flush=True)


def tensor_bytes(qt):
    n = 0
    for f in ("dense", "codes", "codes_packed", "scale", "gw_qsqm", "gw_d",
              "gw_dm", "gw_qh", "gw_qh2", "gw_scale", "gw_min",
              "outlier_indices", "outlier_values"):
        t = getattr(qt, f, None)
        if t is not None:
            n += t.numel() * t.element_size()
    return n


def build(pth, act_stats, label):
    """Квантованная модель из .pth. mmap -- чтобы веса чекпоинта остались
    file-backed страницами и не попали в своп (см. eval_2p9b_one.py)."""
    sd = torch.load(pth, map_location="cpu", mmap=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    n_embd = sd["emb.weight"].shape[1]
    vocab = sd["emb.weight"].shape[0]
    cfg = copy.deepcopy(COMPRESSION)
    cfg.act_stats_path = act_stats
    cfg.bits["small"] = 16          # подтверждено на 1.5B и 2.9B
    t0 = time.time()
    tensors = {k: quantize_tensor(k, w, cfg, real_gw=True) for k, w in sd.items()}
    mb = sum(tensor_bytes(q) for q in tensors.values()) / 1e6
    ckpt = QuantizedCheckpoint(naming="world", n_layer=n_layer, n_embd=n_embd,
                               head_size=64, vocab_size=vocab,
                               tensors=tensors, config_repr=repr(cfg))
    model = QuantRWKV7(ckpt)
    emb_mb = tensor_bytes(ckpt.tensors["emb.weight"]) / 1e6
    print(f"  {label}: L={n_layer} D={n_embd} -> {mb:.1f} MB "
          f"(за токен ~{mb-emb_mb:.0f} MB, emb {emb_mb:.0f} MB -- gather) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    del ckpt, tensors, sd
    gc.collect()
    return model, mb - emb_mb


def argmax_np(lg):
    a = mx.argmax(lg[0], axis=-1)
    mx.eval(a)
    return np.array(a.tolist(), dtype=np.int64)


def acceptance(model_t, model_d, data, langs):
    """Teacher-forcing: доля совпадений argmax на одном и том же тексте.
    Это вероятность принять ПЕРВЫЙ токен раунда (префиксы совпадают по
    построению). Для второго и дальше -- оценка сверху, потому что в бою
    черновик идёт по своему продолжению."""
    at, ad = [], []
    for i in range(data.shape[0]):
        x = mx.array(data[i:i + 1].astype(np.int64))
        at.append(argmax_np(model_t(x)))
        ad.append(argmax_np(model_d(x)))
        mx.clear_cache()
    at, ad = np.stack(at), np.stack(ad)
    m = (at == ad)
    langs = np.array(langs)
    print(f"\n{'язык':<6}{'позиций':>9}{'согласие':>11}"
          + "".join(f"{'E[k='+str(k)+']':>10}" for k in (2, 4, 6)))
    res = {}
    for lang in list(sorted(set(langs.tolist()))) + ["ALL"]:
        sel = slice(None) if lang == "ALL" else (langs == lang)
        p = float(m[sel].mean())
        res[lang] = p
        print(f"{lang:<6}{m[sel].size:>9}{p:>11.4f}"
              + "".join(f"{sum(p**i for i in range(1, k+1))+1:>10.2f}"
                        for k in (2, 4, 6)))
    return res


def gen_plain(model, prompt, n):
    st = model.init_state(1)
    lg, st = model.step(mx.array([prompt]), st, tail_only=1)
    cur = int(argmax_np(lg)[-1])
    out = [cur]
    for _ in range(8):                       # прогрев mx.compile
        lg, _ = model.step(mx.array([[cur]]), st, tail_only=1)
        mx.eval(lg)
    mx.synchronize()
    t0 = time.perf_counter()
    while len(out) < n:
        lg, st = model.step(mx.array([[cur]]), st, tail_only=1)
        cur = int(argmax_np(lg)[-1])
        out.append(cur)
    mx.synchronize()
    return out, (time.perf_counter() - t0) / (len(out) - 1) * 1000


def gen_spec(model_t, model_d, prompt, n, k):
    st, cur = None, None
    lg, st = model_t.step(mx.array([prompt]), model_t.init_state(1), tail_only=1)
    cur = int(argmax_np(lg)[-1])
    lg_d, st_d = model_d.step(mx.array([prompt]), model_d.init_state(1), tail_only=1)

    out, pending, pending_d = [cur], [cur], [cur]
    rounds = acc_tot = drafted = flushes = 0
    mx.synchronize()
    t0 = time.perf_counter()
    while len(out) < n:
        if len(pending) > FLUSH:
            lg, st = model_t.step(mx.array([pending]), st, tail_only=1)
            nxt = int(argmax_np(lg)[-1])
            pending = [nxt]
            pending_d.append(nxt)
            out.append(nxt)
            flushes += 1
            continue

        # черновик: догоняем его state хвостом pending_d, затем k шагов
        lg_d, st_d = model_d.step(mx.array([pending_d]), st_d, tail_only=1)
        pending_d = []
        tokd = int(argmax_np(lg_d)[-1])
        draft = [tokd]
        s = st_d
        for _ in range(k - 1):
            lg_d2, s = model_d.step(mx.array([[tokd]]), s, tail_only=1)
            tokd = int(argmax_np(lg_d2)[-1])
            draft.append(tokd)

        x = pending + draft
        lg, st_new = model_t.step(mx.array([x]), st, tail_only=len(draft) + 1)
        pred = argmax_np(lg)
        m = 0
        while m < len(draft) and draft[m] == int(pred[m]):
            m += 1
        good = draft[:m] + [int(pred[m])]
        out.extend(good)
        pending_d.extend(good)
        rounds += 1
        acc_tot += m
        drafted += len(draft)
        if m == len(draft):
            st = st_new
            pending = [good[-1]]
        else:
            pending = pending + good
    mx.synchronize()
    dt = (time.perf_counter() - t0) / len(out) * 1000
    return out, dt, acc_tot, drafted, rounds, flushes


def main():
    blob = torch.load(CORPUS)
    data, langs = blob["tokens"].numpy(), blob["lang"]
    print(f"K={K}, N_GEN={N_GEN}, корпус {data.shape}", flush=True)
    mem("старт")

    model_d, mb_d = build(DRAFT_PTH, None, "черновик 0.1B")
    model_t, mb_t = build(TARGET_PTH, ACT_STATS, "цель 1.5B")
    mem("обе модели построены")
    print(f"  отношение трафика цель/черновик: {mb_t/mb_d:.1f}x", flush=True)

    accept = acceptance(model_t, model_d, data[:12], langs[:12])

    prompt = data[0, :64].astype(np.int64).tolist()
    sw0 = swap_mb()
    out_p, dt_p = gen_plain(model_t, prompt, N_GEN)
    out_s, dt_s, acc, drafted, rounds, flushes = gen_spec(
        model_t, model_d, prompt, N_GEN, K)
    sw1 = swap_mb()

    print(f"\nplain:      {dt_p:6.2f} мс/ток  ({1000/dt_p:5.1f} ток/с)")
    print(f"spec k={K}:   {dt_s:6.2f} мс/ток  ({1000/dt_s:5.1f} ток/с)  "
          f"-> {dt_p/dt_s:.2f}x")
    print(f"  принято {acc}/{drafted} ({100*acc/max(drafted,1):.0f}%), "
          f"раундов {rounds}, flush {flushes}, "
          f"{(acc+rounds)/max(rounds,1):.2f} ток/раунд")
    same = sum(a == b for a, b in zip(out_p, out_s)) / min(len(out_p), len(out_s))
    print(f"  совпадение выходов plain/spec: {100*same:.0f}%")
    mem("после таймингов")
    if sw1 - sw0 > 5:
        print(f"\n!! своп вырос на {sw1-sw0:.0f} МБ во время замера -- "
              f"цифры скорости НЕДЕЙСТВИТЕЛЬНЫ, перезапустить")
    else:
        print(f"\nсвоп за время замера: {sw1-sw0:+.0f} МБ -- замер валиден")


if __name__ == "__main__":
    main()
