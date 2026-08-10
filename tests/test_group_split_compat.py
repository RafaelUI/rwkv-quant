"""ГЕЙТ: разделение emb_head -> emb + head ничего не изменило для тех,
кто про него не знает.

Псевдоним есть контракт. Пресеты, скрипты в tests/ и конфиги,
восстановленные из манифестов старых .rwkvq, задают `emb_head=N` -- после
разделения они обязаны производить ТЕ ЖЕ БАЙТЫ. Проверяется не глазами:

  1. repr(пресета) не изменился -- он уезжает в манифест как config_repr,
     и его дрейф молча ломает сравнение старых и новых файлов;
  2. quantize_tensor под каждым пресетом даёт бит-в-бит те же буферы
     (сверка по sha256, потоково -- см. make_gw_golden о том, почему не
     тензорами);
  3. псевдоним раскрывается ВО ВСЕХ словарных полях, а не только в bits:
     group_scale, group_scale_mode, clip_percentiles, outlier_fracs;
  4. явный ключ побеждает псевдоним детерминированно, независимо от
     порядка ключей в словаре;
  5. config -> JSON -> config переживает разделение (манифест v2 хранит
     конфиг структурой, и по нему обязано воспроизводиться квантование).

    python tests/make_group_split_golden.py    # на дереве ДО разделения
    python tests/test_group_split_compat.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

GOLDEN = os.environ.get("RWKVQ_SPLIT_GOLDEN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "group_split_golden.json"))
CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))
ROWS = 4096          # срез emb/head: строки в sb6 независимы


def digest(t):
    """sha256 сырых байт. bf16 идёт через .view(int16): numpy его не знает,
    а сверка тут БИТОВАЯ, поэтому приводить к float32 нельзя -- это
    поменяло бы то, что хешируется, и гейт стал бы слабее заявленного."""
    a = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(a.shape)).encode())
    h.update(str(a.dtype).encode())
    if a.dtype == torch.bfloat16:
        a = a.view(torch.int16)
    h.update(a.numpy().tobytes())
    return h.hexdigest()


def iter_preset_outputs():
    """(имя, тензор) для всех буферов, что writer положил бы на диск."""
    from rwkv_quant.presets import PRESETS
    from rwkv_quant.formats import writer
    from rwkv_quant.formats.writer import TENSOR_FIELDS
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    keys = ["emb.weight", "head.weight", "blocks.3.att.key.weight",
            "blocks.3.ffn.value.weight", "blocks.3.att.w1", "blocks.3.att.g1",
            "blocks.3.att.r_k", "blocks.3.att.k_k"]
    for pname, cfg in sorted(PRESETS.items()):
        for key in keys:
            if key not in sd:
                continue
            w = sd[key]
            if w.dim() == 2 and w.shape[0] > ROWS:
                w = w[:ROWS]
            w = w.contiguous()
            for real in (False, True):
                try:
                    qt = writer.quantize_tensor(key, w, cfg, real_gw=real)
                except Exception as e:                      # noqa: BLE001
                    yield f"{pname}|{key}|real={real}|EXC", \
                        torch.tensor([hash(type(e).__name__) % 997])
                    continue
                for f in TENSOR_FIELDS:
                    v = getattr(qt, f, None)
                    if v is None or v.numel() == 0:
                        continue
                    yield f"{pname}|{key}|real={real}|{f}", v
                yield (f"{pname}|{key}|real={real}|meta",
                       torch.tensor([qt.bits, qt.gw_gs, qt.gw_sb]))


def preset_reprs():
    from rwkv_quant.presets import PRESETS
    return {k: repr(v) for k, v in sorted(PRESETS.items())}


def collect():
    return {"reprs": preset_reprs(),
            "hashes": {n: digest(t) for n, t in iter_preset_outputs()}}


def test_golden():
    if not os.path.exists(GOLDEN):
        print(f"эталон {GOLDEN} отсутствует -- снять на дереве ДО разделения:"
              f"\n  git stash && python tests/make_group_split_golden.py && git stash pop")
        return False
    ref = json.load(open(GOLDEN))
    got = collect()
    assert ref["reprs"] == got["reprs"], (
        "repr пресета изменился (он уезжает в манифест как config_repr):\n"
        + "\n".join(f"  {k}\n    было: {ref['reprs'][k]}\n    стало: {got['reprs'].get(k)}"
                    for k in ref["reprs"] if ref["reprs"][k] != got["reprs"].get(k)))
    assert set(ref["hashes"]) == set(got["hashes"]), (
        f"набор буферов разошёлся: пропало "
        f"{sorted(set(ref['hashes']) - set(got['hashes']))[:5]}, появилось "
        f"{sorted(set(got['hashes']) - set(ref['hashes']))[:5]}")
    bad = [n for n in ref["hashes"] if ref["hashes"][n] != got["hashes"][n]]
    assert not bad, f"буферы разошлись ({len(bad)}): {bad[:8]}"
    print(f"1. repr пресетов не изменился: {list(ref['reprs'])}")
    print(f"2. {len(ref['hashes'])} буферов пресетов -- бит-в-бит")
    return True


def test_alias_everywhere():
    from rwkv_quant.calibration.group_config import QuantConfig
    c = QuantConfig(emb_head=5, group_scale={"emb_head": 32},
                    group_scale_mode={"emb_head": "asym_sb6_aw"},
                    clip_percentiles={"emb_head": 99.9},
                    outlier_fracs={"emb_head": 0.01})
    for field in ("bits", "group_scale", "group_scale_mode",
                  "clip_percentiles", "outlier_fracs"):
        d = getattr(c, field)
        assert "emb" in d and "head" in d, f"{field}: псевдоним не раскрыт -> {d}"
        assert "emb_head" not in d, f"{field}: псевдоним протёк наружу -> {d}"
        assert d["emb"] == d["head"], f"{field}: раскрыт неодинаково"
    print("3. псевдоним раскрыт во всех пяти словарных полях")


def test_explicit_wins():
    from rwkv_quant.calibration.group_config import QuantConfig
    a = QuantConfig(emb_head=6, emb=4)
    b = QuantConfig(emb=4, emb_head=6)          # обратный порядок ключей
    assert (a.bits["emb"], a.bits["head"]) == (4, 6), a.bits
    assert (b.bits["emb"], b.bits["head"]) == (4, 6), b.bits
    print("4. явный ключ побеждает псевдоним независимо от порядка")


def test_json_roundtrip():
    from rwkv_quant.formats.writer import config_to_json
    from rwkv_quant.formats.reader import config_from_json
    from rwkv_quant.presets import PRESETS
    for name, cfg in PRESETS.items():
        back = config_from_json(config_to_json(cfg))
        assert back.bits == cfg.bits, f"{name}: bits {back.bits} != {cfg.bits}"
        assert back.group_scale == cfg.group_scale, name
        assert back.group_scale_mode == cfg.group_scale_mode, name
        assert repr(back) == repr(cfg), name
    # и конфиг СТАРОГО манифеста (с ключом emb_head) обязан читаться
    old = {"bits": {"proj": 6, "cmix": 6, "emb_head": 6, "w_lora": 6,
                    "a_lora": 6, "v_lora": 6, "g_lora": 8, "small": 16},
           "group_scale": {"proj": 32, "cmix": 32, "emb_head": 32},
           "group_scale_mode": {"proj": "asym_sb6", "cmix": "asym_sb6_aw",
                                "emb_head": "asym_sb6_aw"},
           "clip_percentiles": {}, "outlier_fracs": {}, "bits_overrides": {},
           "act_stats_path": None}
    c = config_from_json(old)
    assert c.bits["emb"] == c.bits["head"] == 6
    assert c.group_scale["emb"] == c.group_scale["head"] == 32
    assert c.group_scale_mode["head"] == "asym_sb6_aw"
    print("5. config <-> JSON переживает разделение, старый манифест читается")


if __name__ == "__main__":
    ok = test_golden()
    test_alias_everywhere()
    test_explicit_wins()
    test_json_roundtrip()
    print("\nГЕЙТ ЗЕЛЁНЫЙ" + ("" if ok else " (эталон пропущен)"))
