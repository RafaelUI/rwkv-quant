"""Гейт: РЕАЛЬНАЯ упаковка sym (Q6_K) бит-в-бит равна fake-пути.

Утверждение, ради которого гейт существует: файл, произведённый
`real_gw=True`, деквантуется РОВНО в тот тензор, на котором мерили ppl.
Все композитные числа по sym сняты fake-путём, и если упаковщик
разойдётся с ним хоть на бит, перемер на реальном пути покажет другое
число, а объяснить его будет нечем.

Проверяются четыре независимых утверждения, и все требуют РАВЕНСТВА,
а не близости:

  1. writer(real_gw=True) -> reader == writer(real_gw=False).dense
     -- упаковщик против fake-пути;
  2. codec (numpy, нормативный) == reader (torch, быстрый);
  3. роундтрип через контейнер: save_rwkvq -> load_raw -> деквант, плюс
     codec.dequant_key на переоткрытом файле (это путь rwkv-metal и
     SwiftRWKV, у них нет ни torch, ни пакета rwkv_quant);
  4. бит/вес по РЕАЛЬНЫМ размерам буферов, а не по формуле: 8.5625 при
     восьми битах и 6.5625 при шести. Формула могла бы сойтись на бумаге
     и разойтись с файлом -- ровно так была занижена оценка МБ в
     ablate_sym_composite (asym gw64 считался как bits + 64/gs).

Формы берутся из НАСТОЯЩЕГО чекпоинта (закон 17): синтетика, которую
нельзя предъявить в реальном файле, покрытием не является. emb/head
режутся по строкам -- вдоль ВХОДНОЙ оси форма при этом сохраняется
целиком, а именно она определяет раскладку.

    python tests/test_sym_packer_parity.py [ckpt ...]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rwkv_quant.calibration.group_config import QuantConfig  # noqa: E402
from rwkv_quant.formats import codec, writer  # noqa: E402
from rwkv_quant.formats.reader import _dequantize_one  # noqa: E402

CKPTS = sys.argv[1:] or [os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")]
KEYS = ["blocks.3.ffn.key.weight", "blocks.3.ffn.value.weight",
        "blocks.3.att.key.weight", "emb.weight", "head.weight"]
MAX_ROWS = 4096          # emb/head целиком не нужны: раскладка вдоль IN

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          f"{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def _cfg(group, bits, mode, ex2_path=None):
    return QuantConfig(**{group: bits}, group_scale={group: 16},
                       group_scale_mode={group: mode},
                       act_stats_path=ex2_path)


def _np(t):
    return None if t is None else t.detach().cpu().contiguous().numpy()


def codec_dequant(qt):
    """Тот же тензор БЕЗ torch -- нормативный путь потребителей формата."""
    w = codec.dequant_sym(
        _np(qt.gw_qs), _np(qt.gw_d), shape=tuple(qt.shape), gs=qt.gw_gs,
        sb=qt.gw_sb, codes=_np(qt.codes), codes_packed=_np(qt.codes_packed),
        qh=_np(qt.gw_qh), qh2=_np(qt.gw_qh2))
    return torch.from_numpy(w).to(torch.bfloat16)


def _bits_per_weight(qt):
    n = int(qt.shape[0]) * int(qt.shape[1])
    total = 0
    for f in writer.TENSOR_FIELDS:
        v = getattr(qt, f, None)
        if v is not None and v.numel():
            total += v.numel() * v.element_size()
    return total * 8.0 / n


def run_key(sd, key, group, stats_path):
    w = sd[key]
    if w.dim() != 2:
        return
    if w.shape[0] > MAX_ROWS:
        w = w[:MAX_ROWS]
    w = w.contiguous()
    for bits in (8, 6):
        for mode in ("sym_aw", "sym", "sym_plain"):
            cfg = _cfg(group, bits, mode,
                       stats_path if mode.endswith("_aw") else None)
            fake = writer.quantize_tensor(key, w, cfg, real_gw=False)
            real = writer.quantize_tensor(key, w, cfg, real_gw=True)
            tag = f"{key} {tuple(w.shape)} {mode}@{bits}"

            check(f"{tag}: kind", real.gw_mode == "sym"
                  and real.gw_gs == 16 and real.gw_sb == 16,
                  f"gw_mode={real.gw_mode} gs={real.gw_gs} sb={real.gw_sb}")
            # при восьми битах коды -- просто байты, битплоскостей быть
            # не должно вовсе; при шести -- ровно наоборот
            want8 = (real.codes is not None and real.codes_packed is None
                     and real.gw_qh is None and real.gw_qh2 is None)
            want6 = (real.codes is None and real.codes_packed is not None
                     and real.gw_qh is not None and real.gw_qh2 is not None)
            check(f"{tag}: буферы", want8 if bits == 8 else want6)

            deq_r = _dequantize_one(real)
            check(f"{tag}: упаковщик == fake", torch.equal(deq_r, fake.dense),
                  "" if torch.equal(deq_r, fake.dense) else
                  f"max|Δ|={float((deq_r.float()-fake.dense.float()).abs().max()):.3e}, "
                  f"расх. {int((deq_r != fake.dense).sum())}")

            deq_c = codec_dequant(real)
            check(f"{tag}: codec == reader", torch.equal(deq_r, deq_c),
                  "" if torch.equal(deq_r, deq_c) else
                  f"расх. {int((deq_r != deq_c).sum())}")

            bpw = _bits_per_weight(real)
            want = 8.5625 if bits == 8 else 6.5625
            check(f"{tag}: {bpw:.4f} бит/вес", abs(bpw - want) < 1e-9,
                  "" if abs(bpw - want) < 1e-9 else f"ожидалось {want}")
            del fake, real, deq_r, deq_c


def run_container(sd, stats_path, tmp="/tmp/sym_packer_gate.rwkvq"):
    """Роундтрип через контейнер: манифест должен описывать раскладку так,
    чтобы torch-free читалка восстановила её без единой догадки."""
    print("\nконтейнер: save_rwkvq -> load_raw / codec.open_rwkvq")
    from rwkv_quant.formats.reader import load_raw
    from rwkv_quant.formats.schema import QuantizedCheckpoint

    cfg = _cfg("cmix", 8, "sym_aw", stats_path)
    tensors, ref = {}, {}
    for key in ("blocks.3.ffn.key.weight", "blocks.3.ffn.value.weight"):
        w = sd[key].contiguous()
        qt = writer.quantize_tensor(key, w, cfg, real_gw=True)
        tensors[key] = qt
        ref[key] = _dequantize_one(qt)
    ckpt = QuantizedCheckpoint(naming="world", n_layer=1, n_embd=2048,
                               head_size=64, vocab_size=65536,
                               tensors=tensors, config_repr=repr(cfg),
                               config=cfg)
    writer.save_rwkvq(ckpt, tmp)

    back = load_raw(tmp)
    for key, want in ref.items():
        got = _dequantize_one(back.tensors[key])
        check(f"{key}: роундтрип reader", torch.equal(got, want))

    manifest, arrays = codec.open_rwkvq(tmp)
    for key, want in ref.items():
        m = manifest["tensors"][key]
        check(f"{key}: манифест", m["kind"] == "sym"
              and m["n_blocks"] == int(m["shape"][1]) // 16,
              f"kind={m['kind']} n_blocks={m['n_blocks']}")
        got = torch.from_numpy(
            codec.dequant_key(manifest, arrays, key)).to(torch.bfloat16)
        check(f"{key}: роундтрип codec (torch-free)", torch.equal(got, want))
    del arrays
    os.remove(tmp)


def main():
    torch.manual_seed(0)
    for ckpt in CKPTS:
        print(f"\n=== {os.path.basename(ckpt)} ===")
        sd = torch.load(ckpt, map_location="cpu", mmap=True)
        n_embd = int(sd["emb.weight"].shape[1])
        # статистика ОБЯЗАНА принадлежать этому чекпоинту, иначе sym_aw
        # тихо выродится в sym и гейт проверит не тот режим (закон 15)
        stats_path = None
        for cand in sorted(glob.glob("/tmp/act_stats_*.pt")):
            st = torch.load(cand)
            v = st.get("blocks.3.ffn.key.weight")
            if v is not None and int(v.numel()) == n_embd:
                stats_path = cand
                break
        print(f"act_stats: {stats_path or 'НЕТ -- режимы _aw выродятся в sym'}")
        if stats_path is None:
            FAILS.append("act_stats не найдены: режим sym_aw не проверен")
        for key in KEYS:
            if key not in sd:
                continue
            group = ("cmix" if ".ffn." in key else
                     "emb" if key == "emb.weight" else
                     "head" if key == "head.weight" else "proj")
            run_key(sd, key, group, stats_path)
        run_container(sd, stats_path)
        del sd
    print("\nГЕЙТ " + ("ПРОЙДЕН" if not FAILS
                       else f"ПРОВАЛЕН: {len(FAILS)} -- " + "; ".join(FAILS[:5])))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
