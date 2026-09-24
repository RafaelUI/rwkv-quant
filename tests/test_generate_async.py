"""Гейт: generate с конвейером async_eval == синхронный путь ПОБИТНО (24.09).
Утверждения (0.1B по умолчанию, greedy, промпт -- 40 токенов строки eval-корпуса, 64 шага):
  1  sync == sync (повтор): иначе утверждение 2 ничего не значит;
  2  async == sync: токены, логиты каждого шага (хеш байт), итоговый state (хеш байт);
  3  продолжение со state (два вызова по 32) == один вызов на 64, в обоих режимах;
  4  РАЗРЕШЕНИЕ, мутация А: конвейер со state на шаг старше -> ловится;
  5  РАЗРЕШЕНИЕ, мутация Б: выдача следующего токена вместо текущего -> ловится.
    python tests/test_generate_async.py [файл.rwkvq]"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7
from rwkv_quant.backends.metal import generate as G

F = sys.argv[1] if len(sys.argv) > 1 else "~/Develop/WKV-kvant/compression_0p1b_e6_2109.rwkvq"
m = QuantRWKV7(load_raw(os.path.expanduser(F)))
# настоящий текст, а не случайные токены: на случайном промпте greedy уходит в
# период из двух токенов, и сверка ТОКЕНОВ теряет разрешение (логиты и state -- нет)
import torch
P = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt"))["tokens"][int(os.environ.get("RWKVQ_ROW", "3")), :40].tolist()
N = 64


def hbytes(a):
    return hashlib.sha1(np.array(a.astype(mx.float32)).tobytes()).hexdigest()[:16]


def hstate(st):
    h = hashlib.sha1()
    for layer in st:
        for a in layer:
            h.update(np.array(a.astype(mx.float32)).tobytes())
    return h.hexdigest()[:16]


def run(gen_fn, **kw):
    out, toks, lh = [], [], []
    for y, lg in gen_fn(m, kw.pop("prompt", P), kw.pop("n", N), out=out, **kw):
        toks.append(int(y.item())); lh.append(hbytes(lg))
    return toks, lh, hstate(out[0]), out[0]


def mutant(model, prompt, max_tokens, sampler=G.greedy, pipeline=True, state=None, out=None, kind="A"):
    """Копия конвейера generate_step с одной ошибкой."""
    lg, st = G._prefill(model, prompt, state)
    y = sampler(lg)
    prev_st = st

    def one(tok, s):
        l2, s2 = model.step(tok.reshape(1, 1), s)
        l2 = l2[:, -1]
        return sampler(l2), l2, s2
    mx.async_eval(y, lg, st)
    for n in range(max_tokens):
        if n + 1 < max_tokens:
            use = prev_st if (kind == "A" and n >= 2) else st   # А: state на шаг старше
            prev_st = st
            ny, nlg, st = one(y, use)
            mx.async_eval(ny, nlg, st)
        mx.eval(y, lg)
        if kind == "B" and n + 1 < max_tokens:
            mx.eval(ny, nlg)
            yield ny, nlg                                      # Б: сдвиг выдачи
        else:
            yield y, lg
        if n + 1 < max_tokens:
            y, lg = ny, nlg
    mx.eval(st)
    out[:] = [st]


fails = 0
def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print("  %s %s %s" % ("OK  " if ok else "FAIL", name, detail), flush=True)


S1 = run(G.generate_step, pipeline=False)
S2 = run(G.generate_step, pipeline=False)
A1 = run(G.generate_step, pipeline=True)
print("файл %s; токены sync: %s ..." % (os.path.basename(F), S1[0][:12]), flush=True)
check("1 sync == sync", S1[:3] == S2[:3])
check("2 async == sync: токены", A1[0] == S1[0])
check("2 async == sync: логиты всех %d шагов" % N, A1[1] == S1[1],
      "" if A1[1] == S1[1] else "первое расхождение на шаге %d" % next(i for i in range(N) if A1[1][i] != S1[1][i]))
check("2 async == sync: итоговый state", A1[2] == S1[2])
for pipe in (False, True):
    a = run(G.generate_step, pipeline=pipe, n=32)
    b = run(G.generate_step, pipeline=pipe, n=32, prompt=[a[0][-1]], state=a[3])
    check("3 продолжение 32+32 == 64 (%s)" % ("async" if pipe else "sync"),
          a[0] + b[0] == S1[0] and a[1] + b[1] == S1[1] and b[2] == S1[2])
MA = run(lambda *x, **k: mutant(*x, kind="A", **k))
MB = run(lambda *x, **k: mutant(*x, kind="B", **k))
check("4 мутация А (state на шаг старше) поймана", MA[:3] != S1[:3],
      "(токены %s, state %s)" % ("разошлись" if MA[0] != S1[0] else "СОВПАЛИ", "разошёлся" if MA[2] != S1[2] else "СОВПАЛ"))
check("5 мутация Б (сдвиг выдачи) поймана", MB[:2] != S1[:2])
toks, st = G.generate(m, P, N)
check("6 обёртка generate == generate_step", toks == A1[0] and hstate(st) == A1[2])
print("ИТОГ:", "ЗЕЛЁНЫЙ" if fails == 0 else "КРАСНЫЙ (%d)" % fails, flush=True)
sys.exit(1 if fails else 0)
