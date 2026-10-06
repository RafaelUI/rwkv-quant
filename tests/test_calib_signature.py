"""Гейт подписи калибровки (06.10, находка 2 проверки API; решение владельца -- вариант А: хеш токенов окон).

Было: act_stats._signature и autopick._signature брали от токенизатора только type(tokenizer).__name__.
  * один словарь в формах путь / функция / объект / связанный метод -> четыре ключа кеша и разные манифесты;
  * два РАЗНЫХ словаря одной формы -> один ключ: второй молча получал статистику первого.

Свойства (0.1B, кеш -- во временном каталоге, рабочий ~/.cache не трогается):
  S1 один словарь в четырёх формах -> одна подпись act_stats и одна подпись measure;
  S2 разные словари одной формы -> разные подписи (две функции; два пути к разным файлам словаря), и act, и measure;
  S3 сквозное: после сбора одним словарём сбор другим (той же формы) НЕ берёт чужой кеш -- его статистика
     отличается от первой и равна честному пересчёту без кеша;
  S4 подпись зависит от чекпоинта, seq_len и бюджета; стабильна между вызовами и при сброшенной памяти токенизации;
  S5 манифест: quantize() с токенизатором путём и объектом пишет одну и ту же подпись и один act_stats_path;
  S6 кеш, записанный под СТАРЫМ ключом (формула до 06.10), не читается.
Мутации (--mutate): хеш токенов -- константа (S2); в подпись act возвращено имя типа (S1); память токенизации с общим
ключом (S2); подпись measure берёт имя типа вместо токенов (S1).
    python tests/test_calib_signature.py [--mutate]"""
import hashlib, os, sys, tempfile
os.environ.setdefault("RWKVQ_DEVICE", "mps")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from rwkv_quant import quantize, presets
from rwkv_quant.calibration import act_stats as A, autopick as ap
from rwkv_quant.formats import codec

WK = os.path.expanduser("~/Develop/WKV-kvant/")
CK = os.environ.get("CK", WK + "rwkv7-g1d-0.1b.pth")
CK2 = os.environ.get("CK2", WK + "rwkv7-g1d-0.4b-20260210-ctx8192.pth")
TOK = os.environ.get("TOK", "/Users/s/Develop/rwkv-metal/rwkv_metal/tokenizer/rwkv_vocab_v20230424.txt")


def legacy_sig(ckpt, tokenizer, corpus):
    """Формула до 06.10 -- чтобы положить в кеш файл под старым ключом (S6)."""
    h = hashlib.sha256(); st = os.stat(ckpt); h.update(str(st.st_size).encode())
    with open(ckpt, "rb") as f: h.update(f.read(1 << 20))
    with open(corpus, "rb") as f: h.update(f.read())
    h.update(type(tokenizer).__name__.encode())
    return h.hexdigest()[:16]


def props(d):
    R = []
    def check(name, ok, info=""): R.append((name, bool(ok), str(info)[:260]))
    A._TOK_MEMO.clear(); ap._WIN_MEMO.clear()
    cache0 = A.CACHE_DIR; A.CACHE_DIR = os.path.join(d, "cache"); os.makedirs(A.CACHE_DIR)
    try:
        enc = A._encoder(TOK)
        class Obj:
            def encode(self, s): return enc(s)
        def world_fn(s): return enc(s)
        def bytes_fn(s): return list(s.encode("utf-8"))            # другой словарь той же формы (функция)
        tok2 = os.path.join(d, "other_vocab.txt")                    # другой словарь той же формы (путь): мировой без длинных токенов
        lines = open(TOK, encoding="utf-8").read().split("\n")
        open(tok2, "w", encoding="utf-8").write("\n".join(lines[:20000]) + "\n")
        cfg = presets.COMPRESSION
        asig = lambda t: A._signature(CK, t, A.CORPUS)
        msig = lambda t: ap._signature(CK, cfg, A.CORPUS, ap.N_WINDOWS, ap.SEQ_LEN, t)

        forms = {"путь": TOK, "функция": world_fn, "объект": Obj(), "метод": Obj().encode}
        sa = {n: asig(t) for n, t in forms.items()}; sm = {n: msig(t) for n, t in forms.items()}
        check("S1 один словарь в четырёх формах -> одна подпись (act и measure)", len(set(sa.values())) == 1 and len(set(sm.values())) == 1, (sa, sm))

        d_fn = (asig(world_fn) != asig(bytes_fn), msig(world_fn) != msig(bytes_fn))
        d_path = (asig(TOK) != asig(tok2), msig(TOK) != msig(tok2))
        check("S2 разные словари одной формы -> разные подписи (act, measure)", all(d_fn) and all(d_path), ("функции", d_fn, "пути", d_path))

        s1, g1 = A.collect(CK, world_fn, verbose=False)
        s2, g2 = A.collect(CK, bytes_fn, verbose=False)
        s3, _ = A.collect(CK, bytes_fn, verbose=False, cache=False)
        k = sorted(s1)[len(s1) // 2]
        rel = lambda a, b: float((a.float() - b.float()).abs().max() / a.float().abs().max())
        check("S3 сквозное: чужой кеш не подставляется", g1 != g2 and rel(s1[k], s2[k]) > 0.05 and rel(s2[k], s3[k]) < 1e-3,
              "ключи %s / %s; '%s': чужой rel %.2f, честный пересчёт rel %.1e" % (g1, g2, k, rel(s1[k], s2[k]), rel(s2[k], s3[k])))

        base = asig(TOK); A._TOK_MEMO.clear(); again = asig(TOK)
        dep = (A._signature(CK2, TOK, A.CORPUS) != base if os.path.exists(CK2) else None,
               A._signature(CK, TOK, A.CORPUS, seq_len=256) != base, A._signature(CK, TOK, A.CORPUS, budget=1 << 16) != base)   # меньший бюджет окон не меняет: первый круг по чанкам безусловен
        check("S4 зависит от чекпоинта / seq_len / бюджета; стабильна", base == again and all(x for x in dep if x is not None) and dep[0] is not None, (base, again, dep))

        fa, fb = os.path.join(d, "a.rwkvq"), os.path.join(d, "b.rwkvq")
        quantize(CK, fa, preset="compression", tokenizer=TOK, gptq=False, autopick=False, verbose=False)
        quantize(CK, fb, preset="compression", tokenizer=Obj(), gptq=False, autopick=False, verbose=False)
        ma, mb = codec.open_rwkvq(fa)[0], codec.open_rwkvq(fb)[0]
        ca, cb = ma["tokenizer"].split("(calib ")[-1], mb["tokenizer"].split("(calib ")[-1]
        pa, pb = ma["config"]["act_stats_path"], mb["config"]["act_stats_path"]
        check("S5 манифест: путь и объект -> одна подпись и один act_stats_path", ca == cb == base + ")" and pa == pb and base in pa, (ma["tokenizer"], mb["tokenizer"], pa, pb))

        A._TOK_MEMO.clear()
        old = os.path.join(A.CACHE_DIR, "act_%s.pt" % legacy_sig(CK, TOK, A.CORPUS))
        torch.save({"мусор": torch.zeros(1)}, old)
        s6, g6 = A.collect(CK, TOK, verbose=False)
        check("S6 кеш под старым ключом не читается", "мусор" not in s6 and g6 == base and g6 not in old, (os.path.basename(old), g6))
    finally:
        A.CACHE_DIR = cache0; A._TOK_MEMO.clear(); ap._WIN_MEMO.clear()
    return R


def run(label):
    with tempfile.TemporaryDirectory(prefix="rq_sig_gate_") as d:
        R = props(d)
    failed = [n for n, ok, _ in R if not ok]
    for n, ok, info in R:
        print("  [%s] %s%s" % ("OK" if ok else "FAIL", n, "" if ok else " -- " + info))
    print("%s: %d свойств, провалов %d" % (label, len(R), len(failed)), flush=True)
    return failed


def main():
    ok = not run("КОНТРОЛЬ")
    if "--mutate" in sys.argv:
        sig0 = A._signature
        def with_type(ckpt, tokenizer, corpus, *a, **k):
            return hashlib.sha256((sig0(ckpt, tokenizer, corpus, *a, **k) + type(tokenizer).__name__).encode()).hexdigest()[:16]
        muts = [("хеш токенов -- константа", lambda: setattr(A, "tokens_sha", lambda data: "0" * 64), "S2"),
                ("в подпись act возвращено имя типа", lambda: setattr(A, "_signature", with_type), "S1"),
                ("память токенизации с общим ключом", lambda: setattr(A, "_memo_key", lambda t: ("одно",)), "S2"),
                ("подпись measure берёт имя типа", lambda: setattr(ap, "_tokens_term", lambda t, c, s: type(t).__name__), "S1")]
        for name, apply, expect in muts:
            saved = (A.tokens_sha, A._signature, A._memo_key, ap._tokens_term)
            apply()
            try:
                failed = run("МУТАЦИЯ «%s»" % name)
            except Exception as e:
                failed = ["%s (исключение %s: %s)" % (expect, type(e).__name__, str(e)[:80])]; print("  мутация уронила прогон:", failed[0])
            finally:
                A.tokens_sha, A._signature, A._memo_key, ap._tokens_term = saved
            caught = any(f.startswith(expect) for f in failed)
            print("  -> %s (ожидалось падение %s)" % ("ПОЙМАНА" if caught else "НЕ ПОЙМАНА", expect))
            ok &= caught
        ok &= not run("КОНТРОЛЬ ПОСЛЕ МУТАЦИЙ")
    print("ИТОГ: %s" % ("ЗЕЛЁНЫЙ" if ok else "КРАСНЫЙ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
