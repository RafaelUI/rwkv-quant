"""ГЕЙТ ПРАВОК ПАМЯТИ 05.10: полосы хост -> MLX, фьюз r/k/v как владелец буферов (RKV_SHARE), ленивая склейка LoRA.

Все три правки обязаны не менять НИ ОДНОГО БАЙТА выхода, поэтому всюду требуется равенство, а не порог.
Но побитный гейт охраняет поведение, а не правильность, поэтому к равенству добавлены утверждения СВОЙСТВ --
именно их молчаливое невыполнение и было бы настоящей поломкой («флаг включён и ничего не дал»):

  P1  полосы: тензор, собранный _mx_rows, побайтно равен собранному целиком (плотный эмбеддинг sb6, интерлив
      головы и FFN у GwQuantLinear, интерлив 6 бит у SymQuantLinear); полоса нарочно сужена до 1 МиБ, чтобы полос
      было много и на малых матрицах;
  P2  владение: при RKV_SHARE у проекций слитых слоёв НЕТ своих qblk / qsqm|qs / ddm|d, срез читается из фьюза
      и побайтно равен буферу той же проекции в модели без флага;
  P3  поведение: логиты модели с флагом побитно равны логитам модели без флага -- слитый декод, НЕслитый декод
      (FUSE=False: проекции читают срезы на каждом шаге), префилл T=16;
  P4  склейка LoRA: в режиме "sep" её НЕТ ни у одного слоя; при переключении LORA_Q="glue" в рантайме она
      достраивается и выход побитно равен модели, собранной в "glue" с самого начала;
  P5  память: активная память MLX с флагом меньше, чем без него, не менее чем на 90% суммарного размера буферов
      фьюза (иначе вторая копия где-то жива).

РАЗРЕШАЮЩАЯ СПОСОБНОСТЬ МЕРИТСЯ МУТАЦИЕЙ (закон 37): --mutate прогоняет контроль и шесть поломок, каждая
обязана окрасить гейт в красный, и печатает, КАКОЕ утверждение её поймало.

    python tests/test_mem_share_parity.py [файл.rwkvq ...]            # по умолчанию /tmp/champion_v2 и /tmp/reduction_new
    python tests/test_mem_share_parity.py --mutate [файл.rwkvq ...]
"""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.backends.metal import quant_linear_gw as GW  # noqa: E402
from rwkv_quant.backends.metal import quant_linear_sym as SYM  # noqa: E402
from rwkv_quant.backends.metal import quant_model as Q  # noqa: E402
from rwkv_quant.formats.reader import dequantize_banded, load_raw  # noqa: E402

DEFAULT = ["/tmp/champion_v2.rwkvq", "/tmp/reduction_new.rwkvq"]
BUFS = {GW.GwQuantLinear: ("qblk", "qsqm", "ddm"), SYM.SymQuantLinear: ("qblk", "qs", "d")}
IDS = np.random.default_rng(20261005).integers(1, 60000, size=(1, 24)).astype(np.int32)


def nb(a):
    return np.array(a).tobytes()


def fused_blocks(m):
    return [b.tmix for b in m.blocks if getattr(b.tmix, "_rkv_fused", None) is not None]


def build(path, share):
    old = GW.RKV_SHARE
    GW.RKV_SHARE = share
    try:
        m = Q.QuantRWKV7(load_raw(path))
        for b in m.blocks:
            b.tmix._build_fused()
    finally:
        GW.RKV_SHARE = old
    return m


def logits(m):
    """Слитый декод 6 шагов, неслитый декод 6 шагов, префилл T=16 -- без mx.compile (ветка FUSE читается на вызове)."""
    out = {}
    for name, fuse in (("слитый декод", True), ("неслитый декод", False)):
        old = Q.FUSE
        Q.FUSE = fuse
        try:
            st = m.init_state(1)
            acc = []
            for t in range(6):
                lg, st = m.forward_stateful(mx.array(IDS[:, t:t + 1]), st, False)
                mx.eval(lg, st)
                acc.append(nb(lg.astype(mx.float32)))
            out[name] = b"".join(acc)
        finally:
            Q.FUSE = old
    lg, st = m.forward_stateful(mx.array(IDS[:, :16]), m.init_state(1), False)
    mx.eval(lg, st)
    out["префилл T=16"] = nb(lg.astype(mx.float32))
    return out


def check_bands(path, fails):
    """P1. Оракул -- сборка ЦЕЛИКОМ, написанная здесь независимо от _mx_rows."""
    ck = load_raw(path)
    old = GW.HOST_BAND_MB
    GW.HOST_BAND_MB = 1
    n = 0
    try:
        for key in ("emb.weight", "head.weight", "blocks.1.ffn.key.weight", "blocks.1.ffn.value.weight", "blocks.2.att.key.weight"):
            qt = ck.tensors[key]
            mode = getattr(qt, "gw_mode", "")
            if key == "emb.weight":
                if mode != "sb6":
                    continue                                   # sym-эмбеддинг идёт через gather, плотного пути нет
                got = Q._dense(qt)
                ref = dequantize_banded(qt, torch.float16).numpy()
            elif mode == "sb6":
                lin = GW.GwQuantLinear(qt)
                if not lin.__dict__.get("_k3"):
                    continue
                OUT, NB = lin.out_features, lin.NB
                parts = [qt.codes_packed.numpy().reshape(OUT, NB, 16)]
                if qt.gw_qh is not None:
                    parts.append(qt.gw_qh.numpy().reshape(OUT, NB, 4))
                if qt.gw_qh2 is not None:
                    parts.append(qt.gw_qh2.numpy().reshape(OUT, NB, 4))
                got, ref = lin.qblk, np.concatenate(parts, axis=2).reshape(OUT, -1)
            elif mode == "sym" and qt.bits == 6:
                lin = SYM.SymQuantLinear(qt)
                OUT, IN = qt.shape
                NP = IN // 32
                got = lin.qblk
                ref = np.concatenate([qt.codes_packed.numpy().reshape(OUT, NP, 16), qt.gw_qh.numpy().reshape(OUT, NP, 4),
                                      qt.gw_qh2.numpy().reshape(OUT, NP, 4)], axis=2).reshape(OUT, -1)
            else:
                continue
            n += 1
            g = np.array(got)
            if g.shape != ref.shape or g.tobytes() != np.ascontiguousarray(ref).tobytes():
                fails.append("P1 полосы: %s (%s) не равен сборке целиком" % (key, mode))
    finally:
        GW.HOST_BAND_MB = old
    if n < 2:
        fails.append("P1 полосы: проверено тензоров %d -- гейт пуст" % n)
    return n


def check(path, verbose=True):
    fails = []
    n_band = check_bands(path, fails)
    mx.clear_cache()
    gc.collect()
    base = mx.get_active_memory()
    A = build(path, False)
    mx.clear_cache()
    mem_a = mx.get_active_memory() - base
    fa = fused_blocks(A)
    if not fa:
        fails.append("в файле нет ни одного слоя со слитым r/k/v -- гейт пуст")
        return fails
    ref_bytes = {}
    for i, tm in enumerate(fa[:3] + fa[-1:]):
        for nm in ("r_proj", "k_proj", "v_proj"):
            lin = getattr(tm, nm)
            for bname in BUFS[type(lin)]:
                ref_bytes[(tm.layer_id, nm, bname)] = nb(getattr(lin, bname))
    dup = sum(sum(getattr(tm._rkv_fused, bname).nbytes for bname in BUFS[type(tm.r_proj)]) for tm in fa)
    la = logits(A)
    del A, fa, tm, lin
    gc.collect()
    mx.clear_cache()
    base = mx.get_active_memory()
    B = build(path, True)
    mx.clear_cache()
    mem_b = mx.get_active_memory() - base
    fb = fused_blocks(B)
    # P2
    own = 0
    for tm in fb:
        for nm in ("r_proj", "k_proj", "v_proj"):
            lin = getattr(tm, nm)
            if any(bname in vars(lin) for bname in BUFS[type(lin)]) or vars(lin).get("_fz", (None,))[0] is not tm._rkv_fused:
                own += 1
    if own:
        fails.append("P2 владение: у %d проекций из %d остались свои буферы (флаг включён и ничего не дал)" % (own, 3 * len(fb)))
    bad = 0
    for (lid, nm, bname), ref in ref_bytes.items():
        lin = getattr(B.blocks[lid].tmix, nm)
        try:
            got = nb(getattr(lin, bname))
        except Exception as e:  # noqa: BLE001
            got = repr(e).encode()
        bad += got != ref
    if bad:
        fails.append("P2 байты: %d срезов из %d не равны буферу проекции без флага" % (bad, len(ref_bytes)))
    # P3
    try:
        lb = logits(B)
        for k in la:
            if la[k] != lb[k]:
                fails.append("P3 поведение: %s не побитно" % k)
    except Exception as e:  # noqa: BLE001
        fails.append("P3 поведение: исключение %s: %s" % (type(e).__name__, str(e)[:80]))
    # P5
    saved = mem_a - mem_b
    if saved < 0.9 * dup:
        fails.append("P5 память: с флагом меньше на %.0f МиБ, а буферы фьюза -- %.0f МиБ" % (saved / 2**20, dup / 2**20))
    # P4
    if Q.LORA_Q != "sep":
        fails.append("P4: умолчание LORA_Q = %r, гейт написан под 'sep'" % (Q.LORA_Q,))
    else:
        withglue = sum(1 for b in B.blocks if getattr(b.tmix, "_lq_glue", None) is not None)
        built = sum(1 for b in B.blocks if getattr(b.tmix, "_lora_q_built", False))
        if built == 0:
            fails.append("P4: квантованные LoRA не построены ни у одного слоя -- гейт пуст")
        if withglue:
            fails.append("P4 склейка: в режиме sep она построена у %d слоёв" % withglue)
        Q.LORA_Q = "glue"
        try:
            lg1, _ = B.forward_stateful(mx.array(IDS[:, :1]), B.init_state(1), False)
            mx.eval(lg1)
            C = build(path, True)                          # собрана и прогрета уже в режиме glue
            lg2, _ = C.forward_stateful(mx.array(IDS[:, :1]), C.init_state(1), False)
            mx.eval(lg2)
            if nb(lg1) != nb(lg2):
                fails.append("P4 склейка: достроенная при переключении не равна построенной сразу")
            if sum(1 for b in B.blocks if getattr(b.tmix, "_lq_glue", None) is not None) == 0:
                fails.append("P4 склейка: после переключения в glue не построена")
            del C
        except Exception as e:  # noqa: BLE001
            fails.append("P4 склейка: исключение %s: %s" % (type(e).__name__, str(e)[:80]))
        finally:
            Q.LORA_Q = "sep"
    if verbose:
        print("  %s: слитых слоёв %d из %d; полос проверено на %d тензорах; срезов сверено %d; память без флага %.0f МиБ, "
              "с флагом %.0f МиБ (минус %.0f, буферы фьюза %.0f)" % (os.path.basename(path), len(fb), len(B.blocks), n_band,
              len(ref_bytes), mem_a / 2**20, mem_b / 2**20, saved / 2**20, dup / 2**20), flush=True)
    del B
    gc.collect()
    mx.clear_cache()
    return fails


def mutations():
    real_rows = GW._mx_rows

    def rows_reversed(make, OUT, rows):
        if rows >= OUT:
            return real_rows(make, OUT, rows)
        parts = [mx.array(np.ascontiguousarray(make(a, min(a + rows, OUT)))) for a in range(0, OUT, rows)]
        return mx.concatenate(parts[::-1], axis=0)

    def rows_shifted(make, OUT, rows):
        return real_rows(lambda a, b: make(a - 1, b - 1) if a > 0 else make(a, b), OUT, rows)

    def patch_adopt(fn):
        saved = [(c, c._adopt) for c in (GW.GwQuantLinear, SYM.SymQuantLinear)]
        for c, real in saved:
            c._adopt = (lambda real: lambda self, parent, row0: fn(real, self, parent, row0))(real)
        return saved

    def restore(saved):
        for c, real in saved:
            c._adopt = real

    real_lq = Q.QuantTMix._build_lora_q

    def lq_always(self):
        was = getattr(self, "_lora_q_built", False)
        real_lq(self)
        if not was and getattr(self, "_lq_A", None) is not None and self._lq_glue is None:
            self._build_lora_glue()

    yield ("полосы склеены в обратном порядке", lambda: setattr(GW, "_mx_rows", rows_reversed), lambda _: setattr(GW, "_mx_rows", real_rows))
    yield ("полосы после первой сдвинуты на одну строку", lambda: setattr(GW, "_mx_rows", rows_shifted), lambda _: setattr(GW, "_mx_rows", real_rows))
    yield ("срез проекции сдвинут на одну строку", lambda: patch_adopt(lambda real, s, p, r0: real(s, p, r0 + 1 if r0 == 0 else r0 - 1)), restore)
    yield ("проекции r и v читают срезы друг друга", lambda: patch_adopt(lambda real, s, p, r0: real(s, p, p.out_features - s.out_features - r0)), restore)
    yield ("_adopt ничего не делает (флаг включён, копии остались)", lambda: patch_adopt(lambda real, s, p, r0: None), restore)
    yield ("склейка LoRA строится всегда (как до правки)", lambda: setattr(Q.QuantTMix, "_build_lora_q", lq_always), lambda _: setattr(Q.QuantTMix, "_build_lora_q", real_lq))


def main():
    Q.LORA_Q = "sep"   # 06.10: умолчание модуля -- "auto" (намерение файла, у reduction выключено); P4 написан под sep -- принудительно
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or DEFAULT
    mutate = "--mutate" in sys.argv
    ok = True
    for p in paths:
        print("=== %s: контроль ===" % os.path.basename(p), flush=True)
        f = check(p)
        for x in f:
            print("  КРАСНОЕ:", x, flush=True)
        ok &= not f
        if not mutate or f:
            continue
        for name, on, off in mutations():
            tok = on()
            try:
                f = check(p, verbose=False)
            finally:
                off(tok)
            print("  мутация «%s»: %s" % (name, ("ПОЙМАНА -- " + "; ".join(x.split(":")[0] for x in f)) if f else "НЕ ПОЙМАНА"), flush=True)
            ok &= bool(f)
    print("ГЕЙТ ЗЕЛЁНЫЙ" + (" (контроль зелёный, все мутации пойманы)" if mutate else "") if ok else "ГЕЙТ КРАСНЫЙ", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
