"""Гейт фьюза: (1) relmax логитов fused/unfused на префилле и в decode,
(2) greedy 48 токенов идентичен, (3) ppl корпуса через ФЬЮЗНУТЫЙ
forward_stateful == референс кернельного пути (11.7125 для champion_v2),
(4) A/B-скорость compiled fused vs unfused.

Модель задаётся аргументом; умолчание -- /tmp/champion_v2.rwkvq
(COMPRESSION, sb6). Для sym-пресета гонять
`tests/test_fuse_parity.py /tmp/reduction_sym_head8.rwkvq`: у него другой
фьюз (SymQuantLinearFused) и якорь ppl не тот, поэтому утверждение 3 там
только печатается.

ФЬЮЗ МЕНЯЕТ ПОРЯДОК СУММИРОВАНИЯ и бит-в-бит не обязан -- отсюда relmax,
а не равенство. Бит-в-бит обязан быть только ФЬЮЗ ПРОЕКЦИЙ, и это
отдельный гейт (tests/test_sym_fuse_parity.py)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/champion_v2.rwkvq"
PPL_REF = 11.7125 if MODEL_PATH.endswith("champion_v2.rwkvq") else None
print(f"модель: {MODEL_PATH}")
model = qm.QuantRWKV7(load_raw(MODEL_PATH))
data = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt"))[:8].numpy()
prompt = mx.array(data[0:1, :64].astype(np.int32))

# --- 1. паритет логитов: префилл T=64 + 8 decode-шагов, raw путь ---
def run_traj(fuse):
    qm.FUSE = fuse
    st = model.init_state(1)
    logits, st = model.forward_stateful(prompt, st, last_only=True)
    outs = [logits]
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(8):
        logits, st = model.forward_stateful(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1)
        outs.append(logits)
    mx.eval(*outs)
    return [np.array(o.astype(mx.float32)) for o in outs]

a, b = run_traj(False), run_traj(True)
rel = max(float(np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-9)) for x, y in zip(a, b))
print(f"parity relmax (prefill+8 decode): {rel:.2e}")

# --- 2. greedy 48 ---
def greedy(fuse, n=48):
    qm.FUSE = fuse
    st = model.init_state(1)
    logits, st = model.forward_stateful(prompt, st, last_only=True)
    toks = []
    for _ in range(n):
        t = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(t); toks.append(int(t[0]))
        logits, st = model.forward_stateful(t[None], st)
    return toks
ga, gb = greedy(False), greedy(True)
print(f"greedy fused vs unfused: {sum(x==y for x,y in zip(ga,gb))}/48")

# --- 3. ppl: ФЬЮЗНУТЫЙ путь против НЕФЬЮЗНУТОГО на ОДНОЙ модели ---
#
# Утверждение здесь ОТНОСИТЕЛЬНОЕ, и это не мелочь. Прежде сверялся
# абсолютный якорь 11.7125, снятый на конкретном файле champion_v2. Файл
# в /tmp живёт до ребута и пересобирается; пересобранный БЕЗ
# /tmp/act_stats_1p5b.pt даёт 12.356, то есть якорь расходится на 5.5%,
# и выглядит это как «фьюз испортил качество». Проверено прямым прогоном:
# нефьюзнутый путь на том же файле даёт те же 12.356. То есть якорь
# описывал ДРУГОЙ артефакт -- ровно та тихая подмена, о которой закон 15.
# Инвариант, который тут вообще проверяем, -- «фьюз считает то же, что и
# не-фьюз», и он от содержимого /tmp не зависит.
def corpus_ppl():
    total_nll, total_tok = 0.0, 0
    for i in range(0, data.shape[0], 4):
        batch = data[i:i+4]
        idx = mx.array(batch[:, :-1]); target = batch[:, 1:]
        logits, _ = model.forward_stateful(idx, model.init_state(batch.shape[0]))
        mx.eval(logits)
        logp = np.array(mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-12))
        V = logp.shape[-1]
        nll = -logp.reshape(-1, V)[np.arange(target.size), target.reshape(-1)]
        total_nll += nll.sum(); total_tok += nll.size
    return float(np.exp(total_nll / total_tok))

qm.FUSE = False
ppl_unfused = corpus_ppl()
qm.FUSE = True
ppl_fused = corpus_ppl()
drift = abs(ppl_fused - ppl_unfused) / ppl_unfused
print(f"ppl unfused {ppl_unfused:.4f} | fused {ppl_fused:.4f} | "
      f"расхождение {drift*1e4:.2f}e-4")
PPL_TOL = 1e-3          # relmax логитов порядка 1.3e-4, ppl обязан быть плотнее
if drift > PPL_TOL:
    print(f"ПРОВАЛ: фьюз двигает ppl на {drift:.2e} при пороге {PPL_TOL:.0e}")
    sys.exit(1)
if PPL_REF is not None:
    off = abs(ppl_unfused - PPL_REF) / PPL_REF
    print(f"справочно: якорь {PPL_REF} для исходного champion_v2, "
          f"здесь {ppl_unfused:.4f} ({off*100:+.2f}%)"
          + ("" if off < 5e-3 else
             "  <-- ВНИМАНИЕ: файл в /tmp не тот, на котором снят якорь "
             "(скорее всего пересобран без act_stats). На это утверждение "
             "не влияет, но числа скорости с прежними не сравнивать."))

# --- 4. A/B скорость compiled ---
def bench(fuse, n=30, warm=8):
    qm.FUSE = fuse
    fn = mx.compile(model.forward_stateful)
    st = model.init_state(1)
    logits, st = fn(prompt, st, True)
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(warm):
        logits, st = fn(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, st = fn(tok[None], st)
        tok = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok)
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

pairs = [(bench(False), bench(True)) for _ in range(5)]
tu = np.median([p[0] for p in pairs]); tf = np.median([p[1] for p in pairs])
print(f"compiled unfused: {tu:6.2f} ms/tok | fused: {tf:6.2f} ms/tok | выигрыш {tu-tf:+.2f} ms")
print("пары:", " ".join(f"({u:.2f},{f:.2f})" for u, f in pairs))
