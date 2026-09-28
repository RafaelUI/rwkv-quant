"""28.09: источники расширенной калибровки -- ПОТОКОВО, без скачивания датасетов целиком.

Parquet читается через HfFileSystem (HTTP range) ПОСТРАНИЧНО: pre_buffer=False + буферизованный поток + iter_batches
по 16 строк и только нужные столбцы -- row group не поднимается целиком (у UltraData-Code группа ~330 МБ, у qwen ~96 МБ),
на диск -- только итоговые JSONL (единицы МБ на источник). Группы берутся РАВНОМЕРНО по шарду (разнообразие).
JSONL (russian_literature) -- построчно с начала, до нужного числа документов. Нужны только huggingface_hub и pyarrow.
Чаты -> формат RWKV g1: "User: ...\\n\\nAssistant: ..." (<think> сохраняется), повторяющиеся системные подсказки -- выброшены.
    python fetch_calib_sources_2809.py --plan              # что и откуда будет прочитано, без чтения данных
    python fetch_calib_sources_2809.py [--only a,b] [--scale 1.0] [--out ~/Develop/data/calib_src]
Итог: <out>/<имя>.jsonl ({"src","lang","text"}), <out>/manifest.json (репо, sha, файлы, счётчики). Готовые файлы пропускаются.
Лицензии источников разные (см. manifest и README датасетов): набор -- для ЛОКАЛЬНОЙ калибровки, не для раздачи."""
import argparse, json, os, resource, sys, time
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

FW2 = "HuggingFaceFW/fineweb-2"; UD = "openbmb/UltraData-Code"
CONV = "@refs%2Fconvert%2Fparquet"
S = [  # name, repo, файл (или шаблон), вид, n, lang, rg (сколько групп по шарду), доп.
    dict(name="fineweb_edu_en", repo="HuggingFaceFW/fineweb-edu", file="sample/10BT/000_00000.parquet", kind="text", n=240, lang="en", rg=24),
    dict(name="fineweb2_ru", repo=FW2, file="data/rus_Cyrl/train/000_00000.parquet", kind="text", n=200, lang="ru", rg=20),
    dict(name="fineweb2_sr_cyrl", repo=FW2, file="data/srp_Cyrl/train/000_00000.parquet", kind="text", n=120, lang="sr", rg=12),
    dict(name="fineweb2_sr_latn", repo=FW2, file="data/srp_Latn/train/000_00000.parquet", kind="text", n=120, lang="sr_latn", rg=12),
    dict(name="fineweb2_zh", repo=FW2, file="data/cmn_Hani/train/000_00000.parquet", kind="text", n=150, lang="zh", rg=15),
    dict(name="ultradata_l2_py", repo=UD, file="data/UltraData-Code-L2/py/UltraData-Code-L2-py-part-00001-of-00119.parquet", kind="code", n=60, lang="py", rg=3),
    dict(name="ultradata_l2_cpp", repo=UD, file="data/UltraData-Code-L2/cpp/*.parquet", kind="code", n=25, lang="cpp", rg=5),
    dict(name="ultradata_l2_js", repo=UD, file="data/UltraData-Code-L2/js/*.parquet", kind="code", n=25, lang="js", rg=5),
    dict(name="ultradata_l2_rust", repo=UD, file="data/UltraData-Code-L2/rust/*.parquet", kind="code", n=20, lang="rust", rg=4),
    dict(name="ultradata_l2_java", repo=UD, file="data/UltraData-Code-L2/java/*.parquet", kind="code", n=15, lang="java", rg=3),
    dict(name="ultradata_l2_go", repo=UD, file="data/UltraData-Code-L2/go/*.parquet", kind="code", n=15, lang="go", rg=3),
    dict(name="ultradata_l3_py", repo=UD, file="data/UltraData-Code-L3/py/UltraData-Code-L3-py-part-00001-of-00147.parquet", kind="code", n=20, lang="py", rg=3),
    dict(name="qwen38_distill", repo="r0b0tlab/qwen3.8-max-distillation-50k", file="data/train-00000-of-00001.parquet", kind="chat", n=300, lang="en", rg=30),
    dict(name="fable51_lite", repo="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", rev=CONV, file="default/lite/0001.parquet", kind="chat", n=40, lang="en", rg=8, max_chars=16000),
    dict(name="oasst2_ru", repo="OpenAssistant/oasst2", file="data/train-00000-of-00001-88ba0162028a73fc.parquet", kind="oasst", n=150, lang="ru"),
    dict(name="novel_zh", repo="taozi555/novel_text", rev=CONV, file="zh/train/*.parquet", kind="text", n=60, lang="zh", rg=6),
    dict(name="ruslit", repo="RafaelUI/russian_literature", file="ruslit_corpus.jsonl", kind="jsonl", n=60, lang="ru"),
]
MIN = dict(text=1200, code=800, chat=1500, oasst=400, jsonl=4000)
ROLE = dict(system="System", user="User", human="User", prompter="User", assistant="Assistant", gpt="Assistant", tool="Tool", function="Tool")


def chat_text(msgs, max_chars):
    out = []
    for m in msgs or []:
        if isinstance(m, str):             # Fable: элементы -- arrow.json (JSON-строки), 28.09
            try:
                m = json.loads(m)
            except ValueError:
                continue
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or m.get("from") or "").lower()
        if role == "system":
            continue                       # шаблонные системные подсказки -- одинаковые окна
        c = m.get("content") if m.get("content") is not None else m.get("value", "")
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        c = (c or "").strip()
        rc = (m.get("reasoning_content") or "").strip()
        if rc and "<think>" not in c:
            c = "<think>\n%s\n</think>\n%s" % (rc, c)
        tc = m.get("tool_calls")
        if tc:
            c += "\n" + json.dumps(tc, ensure_ascii=False)[:1500]
        if role in ("tool", "function"):
            c = c[:1500]
        if c:
            out.append("%s: %s" % (ROLE.get(role, role.capitalize() or "User"), c))
        if sum(map(len, out)) > max_chars:
            break
    return "\n\n".join(out)[:max_chars]


def rows_text(tbl, s):
    k = s["kind"]
    cols = tbl.column_names
    for r in tbl.to_pylist():
        if k == "code":
            t = r.get("content") or ""
            if r.get("relative_path"):
                t = "// %s\n%s" % (r["relative_path"], t)
        elif k == "chat":
            t = chat_text(r.get("messages") or r.get("conversations"), s.get("max_chars", 20000))
        else:
            t = r.get("text") or r.get("content") or next((r[c] for c in cols if isinstance(r[c], str) and len(r[c]) > 200), "")
        yield t[: s.get("max_chars", 20000)]


def resolve(fs, s):
    p = "datasets/%s%s/%s" % (s["repo"], s.get("rev", ""), s["file"])
    if "*" in p:
        xs = sorted(fs.glob(p))
        if not xs:
            raise FileNotFoundError(p)
        p = xs[0]
    return p


def fetch_parquet(fs, s, p):
    got = []
    with fs.open(p, "rb", block_size=2 << 20) as h:
        pf = pq.ParquetFile(h, pre_buffer=False, buffer_size=1 << 20)
        names = pf.schema_arrow.names
        want = dict(code=["content", "relative_path"], chat=["messages", "conversations"]).get(s["kind"], ["text", "content"])
        cols = [c for c in want if c in names] or None
        G = pf.metadata.num_row_groups
        k = max(1, min(G, s.get("rg", 10)))
        per = -(-s["n"] // k)
        for gi in sorted(set(int(i * G / k) for i in range(k))):
            took = 0
            for b in pf.iter_batches(batch_size=16, row_groups=[gi], columns=cols):
                for t in rows_text(pa.Table.from_batches([b]), s):
                    if len(t) >= MIN[s["kind"]]:
                        got.append(t); took += 1
                    if took >= per or len(got) >= s["n"]:
                        break
                if took >= per or len(got) >= s["n"]:
                    break
            if len(got) >= s["n"]:
                break
    return got


def fetch_oasst(fs, s, p):
    """пары запрос -> ответ на языке s["lang"] (дерево сообщений: parent_message_id)."""
    got, prom = [], {}
    with fs.open(p, "rb", block_size=8 << 20) as h:
        pf = pq.ParquetFile(h)
        for gi in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(gi, columns=["message_id", "parent_id", "text", "role", "lang"]).to_pylist():
                if r["lang"] != s["lang"]:
                    continue
                if r["role"] == "prompter":
                    prom[r["message_id"]] = r["text"]
                elif r["role"] == "assistant" and r["parent_id"] in prom:
                    t = "User: %s\n\nAssistant: %s" % (prom.pop(r["parent_id"]).strip(), r["text"].strip())
                    if len(t) >= MIN["oasst"]:
                        got.append(t)
                if len(got) >= s["n"]:
                    return got
    return got


def fetch_jsonl(fs, s, p):
    got = []
    with fs.open(p, "r", encoding="utf-8", block_size=4 << 20) as h:
        for line in h:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("text") or r.get("content") or next((v for v in r.values() if isinstance(v, str) and len(v) > 200), "")
            if len(t) >= MIN["jsonl"]:
                a = len(t) // 3                    # середина книги, без титулов и предисловий
                got.append(t[a:a + s.get("max_chars", 12000)])
            if len(got) >= s["n"]:
                break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/Develop/data/calib_src")
    ap.add_argument("--only", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    fs, api = HfFileSystem(), HfApi()
    only = set(x for x in a.only.split(",") if x)
    mpath = os.path.join(out, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for s in S:
        if only and s["name"] not in only:
            continue
        s = dict(s, n=max(1, int(s["n"] * a.scale)))
        dst = os.path.join(out, s["name"] + ".jsonl")
        try:
            p = resolve(fs, s)
            size = fs.info(p).get("size")
        except Exception as e:
            print("%-18s ОШИБКА пути: %s" % (s["name"], str(e)[:150]), flush=True); continue
        if a.plan:
            print("%-18s %-7s n=%-4d %s (%.2f ГБ файл; читается выборочно)" % (s["name"], s["kind"], s["n"], p.replace("datasets/", ""), (size or 0) / 1e9), flush=True)
            continue
        if os.path.exists(dst):
            print("%-18s уже есть -- пропуск" % s["name"], flush=True); continue
        t0 = time.time()
        try:
            fn = dict(oasst=fetch_oasst, jsonl=fetch_jsonl).get(s["kind"], fetch_parquet)
            got = fn(fs, s, p)
        except Exception as e:
            print("%-18s ОШИБКА чтения: %s: %s" % (s["name"], type(e).__name__, str(e)[:200]), flush=True); continue
        with open(dst + ".tmp", "w", encoding="utf-8") as f:
            for t in got:
                f.write(json.dumps(dict(src=s["name"], lang=s["lang"], text=t), ensure_ascii=False) + "\n")
        os.replace(dst + ".tmp", dst)
        try:
            sha = api.dataset_info(s["repo"]).sha
        except Exception:
            sha = None
        man[s["name"]] = dict(repo=s["repo"], file=p.replace("datasets/", ""), sha=sha, docs=len(got), chars=sum(map(len, got)),
                              lang=s["lang"], kind=s["kind"], fetched=time.strftime("%Y-%m-%d %H:%M"))
        json.dump(man, open(mpath, "w"), ensure_ascii=False, indent=1)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
        print("%-18s %4d док., %.1f МБ текста, %.0f с, пик памяти процесса %.0f МБ" % (s["name"], len(got), sum(map(len, got)) / 1e6, time.time() - t0, rss), flush=True)


if __name__ == "__main__":
    main()
