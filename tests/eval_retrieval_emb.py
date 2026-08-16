"""ЗАДАЧНАЯ МЕТРИКА ДЛЯ REDUCTION: переживает ли РАНЖИРОВАНИЕ квантование.

ЗАЧЕМ. REDUCTION существует ради QLoRA и ВЕКТОРНЫХ моделей, но всё
качество до сих пор меряно ppl и KL -- то есть по логитам, которых у
векторной модели нет вовсе. Здесь метрика прямая: остаётся ли
релевантный документ ближе к запросу, чем нерелевантный, и на сколько
сжимается зазор.

ЭТО ЖЕ ПРОВЕРКА ТОГО, ЧТО МЫ КУПИЛИ. Бит для emb и proj доплачены по KL,
потому что для векторных моделей верность распределения важнее
вероятности одного токена. Утверждение правдоподобное, но непроверенное:
что доли процента ppl и KL значат для ранжирования, не мерил никто.

Модель: RWKV-7 0.4B, дообученная под семантический поиск (C=1024,
24 слоя). Рецепт эмбеддинга взят из её кода (EmbeddingRWKV,
sft_curriculum/src/dataset.py): к тексту дописывается ОДИН EOS (65535),
эмбеддинг -- скрытое состояние на его позиции после ln_out,
L2-нормированное. Длина фиксирована, лишнее обрезается СЛЕВА (как у
них), хвост добивается нулями -- в причинной модели токены ПОСЛЕ EOS на
его позицию не влияют, поэтому паддинг честен и даёт одну форму на все
батчи (иначе mx.compile трассировал бы каждую длину заново).

Данные: LitRetrieval (ru/en), строки задачи `retrieval`: anchor -- запрос
с инструкцией, positive -- релевантный отрывок, negative -- нерелевантный.

    python tests/eval_retrieval_emb.py <файл.rwkvq|bf16> [N] [--ref путь]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rwkv_quant.backends.metal.quant_model as qm  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402

WHAT = sys.argv[1] if len(sys.argv) > 1 else "bf16"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
CKPT = os.environ.get("RWKVQ_EMB_CKPT", "/tmp/emb04b_world.pth")
DATA = os.path.expanduser("~/Develop/retrieval_literature/train.jsonl")
VOCAB = os.path.expanduser(
    "~/Develop/EmbeddingRWKV/embedding/eval/src/reference/rwkv_vocab_v20230424.txt")
OUTDIR = os.environ.get("RWKVQ_EMB_OUT", "/tmp/emb_retrieval")
L = int(os.environ.get("RWKVQ_EMB_LEN", 256))
B = int(os.environ.get("RWKVQ_EMB_BATCH", 16))
EOS = 65535
BOOT = 20000


def swap_mb():
    env = dict(os.environ, LC_ALL="C", LANG="C")
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], env=env,
                         capture_output=True, text=True).stdout
    parts = out.replace("=", " ").split()
    try:
        return float(parts[parts.index("used") + 1].rstrip("M").replace(",", "."))
    except (ValueError, IndexError):
        return float("nan")


def load_triplets(n):
    sys.path.insert(0, os.path.expanduser("~/Develop/WKV-kvant"))
    from world_tokenizer import RWKV_WORLD_TOKENIZER
    tok = RWKV_WORLD_TOKENIZER(VOCAB)
    out = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") != "retrieval":
                continue
            out.append((r["anchor"], r["positive"], r["negative"]))
            if len(out) >= n:
                break
    ids = np.zeros((len(out), 3, L), dtype=np.int32)
    pos = np.zeros((len(out), 3), dtype=np.int32)
    for i, tri in enumerate(out):
        for j, text in enumerate(tri):
            t = list(tok.encode(text))[-(L - 1):]     # обрезка СЛЕВА, как у них
            ids[i, j, :len(t)] = t
            ids[i, j, len(t)] = EOS
            pos[i, j] = len(t)
    return ids, pos, len(out)


def build_model(what):
    if what.endswith(".rwkvq"):
        return qm.QuantRWKV7(load_raw(what))
    # bf16: тот же путь сборки, но без квантования -- значит разница с
    # квантованными конфигами это РОВНО квантование, а не другой код
    from rwkv_quant.calibration.group_config import QuantConfig
    from rwkv_quant.formats.schema import QuantizedCheckpoint
    from rwkv_quant.formats.writer import quantize_tensor
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    cfg_override = None
    if what != "bf16":
        # имя конфига из ablate_sym_composite: пресеты применяются к ЭТОМУ
        # чекпоинту, а статистика активаций обязана быть ЕГО (закон 15 --
        # чужая молча выродит AW в обычный поиск, и замер пройдёт не тот,
        # что заказан)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ablate_sym_composite as comp
        if what not in comp.CONFIGS:
            raise SystemExit(f"не знаю {what}; есть: {list(comp.CONFIGS)}")
        cfg_override = comp.CONFIGS[what]()
        cfg_override.act_stats_path = os.environ.get("RWKVQ_EMB_ACT") or None
        print(f"конфиг {what}, act_stats={cfg_override.act_stats_path}",
              flush=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    emb, r_k = sd["emb.weight"], next(v for k, v in sd.items()
                                      if k.endswith("r_k"))
    cfg = cfg_override if cfg_override is not None else QuantConfig()
    ck = QuantizedCheckpoint(
        naming="world", n_layer=n_layer, n_embd=int(emb.shape[1]),
        vocab_size=int(emb.shape[0]), head_size=int(r_k.shape[-1]),
        config_repr=repr(cfg),
        tensors={k: quantize_tensor(k, w, cfg, real_gw=True)
                 for k, w in sd.items()})
    return qm.QuantRWKV7(ck)


def embed(model, ids, pos):
    """[n, 3, L] -> [n, 3, D], нормированные."""
    n = ids.shape[0]
    flat = ids.reshape(-1, L)
    fpos = pos.reshape(-1)
    D = model.n_embd
    out = np.zeros((flat.shape[0], D), dtype=np.float32)
    t0 = time.time()
    for i in range(0, flat.shape[0], B):
        chunk = flat[i:i + B]
        st = model.init_state(chunk.shape[0])
        h, _ = model.forward_hidden(mx.array(chunk), st)
        h = np.array(h.astype(mx.float32))
        for j in range(chunk.shape[0]):
            out[i + j] = h[j, fpos[i + j]]
        del h
        if i and i % (B * 50) == 0:
            print(f"    {i}/{flat.shape[0]}  ({time.time()-t0:.0f} с)", flush=True)
    out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9
    return out.reshape(n, 3, D)


def metrics(e):
    a, p, ng = e[:, 0], e[:, 1], e[:, 2]
    sp = (a * p).sum(-1)
    sn = (a * ng).sum(-1)
    return sp, sn


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    sw0 = swap_mb()
    ids, pos, n = load_triplets(N)
    print(f"троек retrieval: {n}, длина {L}, батч {B}, "
          f"медиана позиции EOS {np.median(pos):.0f}", flush=True)
    model = build_model(WHAT)
    print(f"модель {WHAT} собрана, своп {sw0:.0f} -> {swap_mb():.0f} МБ",
          flush=True)
    t0 = time.time()
    e = embed(model, ids, pos)
    sp, sn = metrics(e)
    acc = float((sp > sn).mean())
    print(f"\n{WHAT}: точность {acc*100:.2f}%  зазор {float((sp-sn).mean()):.4f}"
          f"  cos+ {float(sp.mean()):.4f}  cos- {float(sn.mean()):.4f}"
          f"  [{time.time()-t0:.0f} с]")
    tag = os.path.basename(WHAT).replace(".rwkvq", "")
    np.savez(f"{OUTDIR}/{tag}.npz", emb=e, sp=sp, sn=sn)
    print(f"-> {OUTDIR}/{tag}.npz")

    ref = f"{OUTDIR}/bf16.npz"
    if tag != "bf16" and os.path.exists(ref):
        r = np.load(ref)
        rsp, rsn = r["sp"], r["sn"]
        drift = 1 - (e * r["emb"]).sum(-1)          # 1 - cos с эталоном
        rng = np.random.default_rng(20260816)
        idx = rng.integers(0, n, size=(BOOT, n))
        d_acc = np.array([((sp[j] > sn[j]).mean()
                           - (rsp[j] > rsn[j]).mean()) * 100 for j in idx])
        d_mar = np.array([((sp[j] - sn[j]).mean()
                           - (rsp[j] - rsn[j]).mean()) for j in idx])
        print(f"\nпротив bf16 (парный бутстрэп по тройкам, {BOOT} итераций):")
        print(f"  точность {float((rsp>rsn).mean())*100:.2f}% -> {acc*100:.2f}% "
              f"({(acc-float((rsp>rsn).mean()))*100:+.2f} п.п., "
              f"95% CI [{np.percentile(d_acc,2.5):+.2f}; "
              f"{np.percentile(d_acc,97.5):+.2f}])")
        print(f"  зазор {float((rsp-rsn).mean()):.4f} -> "
              f"{float((sp-sn).mean()):.4f} "
              f"({float((sp-sn).mean())-float((rsp-rsn).mean()):+.4f}, "
              f"95% CI [{np.percentile(d_mar,2.5):+.4f}; "
              f"{np.percentile(d_mar,97.5):+.4f}])")
        print(f"  дрейф самих векторов: 1-cos = {drift.mean():.2e} "
              f"(макс {drift.max():.2e})")
    print(f"своп: {sw0:.0f} -> {swap_mb():.0f} МБ")


if __name__ == "__main__":
    main()
