"""04.10: отложенные наборы статьи -> tests/data/*.json (токены словаря rwkv_vocab_v20230424, без pickle).
    python pt2json_eval_0410.py"""
import json, os, torch
W = os.path.expanduser("~/Develop/WKV-kvant/"); O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(O, exist_ok=True)
for name in ("eval_code_open", "eval_text_heldout"):
    d = torch.load(W + name + ".pt", weights_only=False)
    out = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
    out["vocab"] = "rwkv_vocab_v20230424"
    p = os.path.join(O, name + ".json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    back = json.load(open(p, encoding="utf-8"))
    assert torch.equal(torch.tensor(back["tokens"]), d["tokens"]), name
    print(name, {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items() if k != "tokens"}, tuple(d["tokens"].shape), os.path.getsize(p))
