"""Генерация по .rwkvq с конвейером async_eval (24.09).

Зачем. На шаге декода RWKV-7 ~1400 запусков (1.5B), и при mx.eval на каждом
токене GPU простаивает, пока CPU строит и кодирует граф следующего шага. Конвейер
(как generate_step в mlx_lm): граф токена t+1 строится и ставится в очередь
async_eval, ПОКА GPU считает токен t; ждём только готовности t. Замер 22.09
(2.9B, ABBA против Qwen): -11% мс/ток. Числа меняются, а не порядок: граф и
его входы те же, поэтому траектория, логиты и state обязаны совпасть с
синхронным путём ПОБИТНО -- гейт tests/test_generate_async.py.

    from rwkv_quant.backends.metal.generate import generate_step, generate
    for tok, logits in generate_step(model, prompt_ids, 128):
        ...                      # tok: mx.array [1] int32, уже вычислен
    toks, state = generate(model, prompt_ids, 128)

sampler: callable(logits [1, V]) -> mx.array [1]; по умолчанию greedy (argmax).
state: продолжить с готового состояния (например, после предыдущего вызова);
тогда prompt -- токены, которых модель ещё не видела.
Цена конвейера: при досрочной остановке (стоп-токен) один лишний шаг уже в
очереди -- он просто выбрасывается.
"""
import mlx.core as mx


def greedy(logits):
    return mx.argmax(logits, axis=-1)


def _prefill(model, prompt, state):
    p = mx.array(prompt, dtype=mx.int32).reshape(1, -1)
    if p.shape[1] < 1:
        raise ValueError("prompt пуст: нужен хотя бы один токен")
    st = model.init_state(1) if state is None else state
    if p.shape[1] > 1:
        # last_only: голова только по последней позиции; T=1 идёт без флага,
        # чтобы не заводить второй кеш трассировки под тот же декодный граф
        lg, st = model.step(p, st, True)
    else:
        lg, st = model.step(p, st)
    return lg[:, -1], st


def generate_step(model, prompt, max_tokens, sampler=greedy, pipeline=True, state=None, out=None):
    """Генератор (tok, logits). Если передан список out, в out[0] по окончании
    кладётся state ПОСЛЕ всех поданных токенов, кроме последнего выданного
    (одинаково в обоих режимах)."""
    if max_tokens <= 0:
        return
    step = model.step
    lg, st = _prefill(model, prompt, state)
    y = sampler(lg)

    def one(tok, st):
        l2, s2 = step(tok.reshape(1, 1), st)
        l2 = l2[:, -1]
        return sampler(l2), l2, s2

    if not pipeline:
        for n in range(max_tokens):
            mx.eval(y, lg, st)
            yield y, lg
            if n + 1 < max_tokens:
                y, lg, st = one(y, st)
    else:
        mx.async_eval(y, lg, st)
        for n in range(max_tokens):
            if n + 1 < max_tokens:
                ny, nlg, st = one(y, st)
                mx.async_eval(ny, nlg, st)
            mx.eval(y, lg)
            yield y, lg
            if n + 1 < max_tokens:
                y, lg = ny, nlg
    if out is not None:
        mx.eval(st)
        out[:] = [st]


def generate(model, prompt, max_tokens, sampler=greedy, pipeline=True, state=None, stop=None):
    """Список токенов (int) и state после них (кроме последнего выданного).
    stop: множество токенов, на которых остановиться (стоп-токен входит в список)."""
    out, toks = [], []
    for y, _ in generate_step(model, prompt, max_tokens, sampler, pipeline, state, out):
        t = int(y.item())
        toks.append(t)
        if stop and t in stop:
            break
    return toks, (out[0] if out else None)
