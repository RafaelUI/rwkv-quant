"""Гейт намерения lora_q (06.10, решение владельца по находке 5 проверки API).

Квантованные LoRA на декоде (quant_model.LORA_Q) -- ДРУГОЕ квантование, чем записано в файле: декод T=1 считал одними
весами, префилл -- другими. Замер прибором статьи: у reduction это +3-6% KL к bf16 на 0.1B-2.9B. Теперь это намерение
файла, как fast_ln: compression -- да, reduction -- нет.

Свойства (0.1B; свежие сборки без GPTQ и файлы files_0110, собранные ДО появления поля):
  L1 пресеты несут намерение, свежие файлы пишут его в манифест (reduction: lora_q False, compression: True);
  L2 рантайм читает: свежие -- из манифеста; старые файлы без поля -- reduction выключено, compression включено;
     явный аргумент QuantRWKV7(..., lora_q=) перекрывает в обе стороны;
  L3 ГЛАВНОЕ: reduction -- декод по одному токену равен префиллу В ПРЕДЕЛАХ ПОЛА (KL к цельному префиллу не больше
     5x KL перенарезки самого префилла), на свежем и на старом файле; квантованные буферы LoRA при этом не строятся;
  L4 прибор не слеп: тот же reduction-файл с lora_q=True уходит от префилла больше чем на 10 полов;
  L5 compression -- намерение сохранено: декод с квантованными LoRA (больше 10 полов), а с lora_q=False -- в полу;
  L6 явный модульный LORA_Q принудителен, как раньше: "sep" включает квантованные ветки на reduction-модели,
     None выключает на compression-модели; "auto" возвращает намерение.
Мутации (--mutate): намерение модели игнорируется, всегда sep (L3); REDUCTION.runtime_lora_q = True (L1); рантайм читает
только манифест, без пресета и fast_ln (L2: старые файлы); модульный флаг не принудителен (L6).
    python tests/test_lora_q_intent.py [--mutate]"""
import copy, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
import mlx.core as mx
from rwkv_quant import quantize, presets
from rwkv_quant.formats import codec
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

WK = os.path.expanduser("~/Develop/WKV-kvant/")
CK = os.environ.get("CK", WK + "rwkv7-g1d-0.1b.pth")
TOK = os.environ.get("TOK", "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")
OLD_RED, OLD_COMP = WK + "files_0110/0p1b_reduction.rwkvq", WK + "files_0110/0p1b_compression.rwkvq"
IDS = torch.load(WK + "eval_text_heldout.pt", weights_only=False)["tokens"][12, :192].numpy().astype(np.int32)[None]
P0, T = 128, 192


def logits(m, splits):
    st = m.init_state(1); out = []; p = 0
    for s in splits:
        lg, st = m.step(mx.array(IDS[:, p:p + s]), st); lg = lg.astype(mx.float32); mx.eval(lg, st)
        out.append(np.array(lg)[0]); p += s
    return torch.log_softmax(torch.from_numpy(np.concatenate(out, 0)[P0:]).double(), -1)


def drift(m):
    """(KL пола, KL декода) к цельному префиллу на позициях 128..191."""
    full = logits(m, [T])
    kl = lambda lq: float((full.exp() * (full - lq)).sum(-1).mean())
    return kl(logits(m, [P0] + [2] * ((T - P0) // 2))), kl(logits(m, [P0] + [1] * (T - P0)))


def props(d):
    R = []
    def check(name, ok, info=""): R.append((name, bool(ok), str(info)[:220]))
    assert qm.LORA_Q == "auto", qm.LORA_Q
    fr, fc = os.path.join(d, "red.rwkvq"), os.path.join(d, "comp.rwkvq")
    quantize(CK, fr, preset="reduction", tokenizer=TOK, gptq=False, verbose=False)
    quantize(CK, fc, preset="compression", tokenizer=TOK, gptq=False, autopick=False, verbose=False)
    rt = lambda p: codec.open_rwkvq(p)[0].get("runtime")

    check("L1 намерение пресетов и манифест свежих файлов",
          presets.REDUCTION.runtime_lora_q is False and presets.COMPRESSION.runtime_lora_q is True
          and rt(fr) == {"fast_ln": False, "lora_q": False} and rt(fc) == {"fast_ln": True, "lora_q": True}, (rt(fr), rt(fc)))

    M = {n: qm.QuantRWKV7(load_raw(p)) for n, p in (("red", fr), ("comp", fc), ("old_red", OLD_RED), ("old_comp", OLD_COMP))}
    src = {n: (m.lora_q, m.lora_q_source) for n, m in M.items()}
    assert "lora_q" not in (rt(OLD_RED) or {}) and "lora_q" not in (rt(OLD_COMP) or {}), "старые файлы должны быть без поля"
    forced = (qm.QuantRWKV7(load_raw(fr), lora_q=True).lora_q, qm.QuantRWKV7(load_raw(fc), lora_q=False).lora_q)
    # файл 22.09-06.10 с правленым с тех пор пресетом: preset_of его теряет, остаётся только fast_ln=False
    class _Old: runtime = {"fast_ln": False}; config_repr = "QuantConfig(не пресет)"
    class _Own: runtime = None; config_repr = "QuantConfig(свой)"
    fb = (qm.resolve_lora_q(_Old(), None, None)[0], qm.resolve_lora_q(_Own(), None, None)[0])
    check("L2 рантайм читает намерение (свежие -- манифест, старые -- без поля), аргумент перекрывает",
          src["red"] == (False, "манифест") and src["comp"] == (True, "манифест") and src["old_red"][0] is False
          and src["old_comp"][0] is True and forced == (True, False) and fb == (False, True), (src, forced, fb))

    D = {n: drift(m) for n, m in M.items()}
    ok3 = all(D[n][1] <= 5 * D[n][0] for n in ("red", "old_red"))
    nobuf = all(all(getattr(b.tmix, "_lq_A", None) is None for b in M[n].blocks) for n in ("red", "old_red"))
    check("L3 reduction: декод == префилл в пределах пола, буферы LoRA не строятся", ok3 and nobuf,
          "KL пол / декод: свежий %.2e / %.2e, старый %.2e / %.2e; буферов нет: %s" % (*D["red"], *D["old_red"], nobuf))

    mon = qm.QuantRWKV7(load_raw(fr), lora_q=True); don = drift(mon)
    check("L4 прибор не слеп: reduction с lora_q=True уходит от префилла", don[1] > 10 * don[0], "%.2e / %.2e" % don)

    moff = qm.QuantRWKV7(load_raw(fc), lora_q=False); doff = drift(moff)
    check("L5 compression: декод с квантованными LoRA; с lora_q=False -- в полу",
          all(D[n][1] > 10 * D[n][0] for n in ("comp", "old_comp")) and doff[1] <= 5 * doff[0],
          "свежий %.2e / %.2e, старый %.2e / %.2e, выключено %.2e / %.2e" % (*D["comp"], *D["old_comp"], *doff))

    try:
        qm.LORA_Q = "sep"; a = drift(qm.QuantRWKV7(load_raw(fr)))
        qm.LORA_Q = None; b = drift(qm.QuantRWKV7(load_raw(fc)))
        qm.LORA_Q = "auto"; c = drift(qm.QuantRWKV7(load_raw(fr)))
    finally:
        qm.LORA_Q = "auto"
    check("L6 явный модульный LORA_Q принудителен, auto возвращает намерение",
          a[1] > 10 * a[0] and b[1] <= 5 * b[0] and c[1] <= 5 * c[0], "sep на reduction %.2e / %.2e; None на compression %.2e / %.2e; auto %.2e / %.2e" % (*a, *b, *c))
    return R


def run(label):
    with tempfile.TemporaryDirectory() as d:
        R = props(d)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s -- %s" % ("OK" if ok else "FAIL", n, info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        TM = qm.QuantTMix
        def manifest_only(ckpt, arg=None, preset=None):
            rt = getattr(ckpt, "runtime", None) or {}
            return (bool(arg), "аргумент") if arg is not None else ((bool(rt["lora_q"]), "манифест") if "lora_q" in rt else (True, "умолчание"))
        muts = [("намерение модели игнорируется (всегда sep)", lambda: setattr(TM, "_lq_mode", lambda self: "sep"), "L3"),
                ("REDUCTION.runtime_lora_q = True", lambda: setattr(presets.REDUCTION, "runtime_lora_q", True), "L1"),
                ("рантайм читает только манифест", lambda: setattr(qm, "resolve_lora_q", manifest_only), "L2"),
                ("модульный флаг не принудителен", lambda: setattr(TM, "_lq_mode", lambda self: "sep" if self.lora_q_on else None), "L6")]
        for name, apply, expect in muts:
            saved = (TM._lq_mode, presets.REDUCTION.runtime_lora_q, qm.resolve_lora_q)
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            finally:
                TM._lq_mode, presets.REDUCTION.runtime_lora_q, qm.resolve_lora_q = saved
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
