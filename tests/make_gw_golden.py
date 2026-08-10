"""Замораживает ХЕШИ выходов блочных fake-quant функций — эталон для
tests/test_groupwise_parity.py.

ПОЧЕМУ ХЕШИ, А НЕ ТЕНЗОРЫ. Первая версия складывала в .pt сами выходы:
~30 вариантов на случай, и в списке случаев были emb.weight и
head.weight (65536x768 = 50.3M элементов, 201 МБ в fp32 каждый). Вышел
файл на 10.2 ГБ, а гейт держал `torch.load(эталон)` И пересчёт
ОДНОВРЕМЕННО — 22 ГБ на машине с 16. Своп, паника, замер убит.

Мораль шире случая и просится в законы: гейт, который сверяет эталон с
пересчётом, стоит УДВОЕННОЙ памяти самого большого набора, если сверять
его целиком. Для бит-в-бит проверки этого не нужно — достаточно хеша
сырых байт: любое расхождение хотя бы в одном бите меняет дайджест.
Теряется только «где именно», и это возвращается пересчётом одного
упавшего случая (см. --explain в гейте).

Здесь всё потоково: одновременно жив ровно один выходной тензор, эталон
— несколько КБ JSON. Перед началом и в конце печатается своп (закон 11).

    python tests/make_gw_golden.py            # записать эталон
    python tests/test_groupwise_parity.py      # сверить
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

GOLDEN = os.environ.get("RWKVQ_GW_GOLDEN",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "fixtures", "gw_golden.json"))
CKPT = os.environ.get("RWKVQ_CKPT", os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1d-0.1b.pth"))

# Ни один случай не должен требовать больше этого — иначе гейт становится
# тем, чем он уже один раз стал. Блочный поиск держит порядка десяти
# копий блока, отсюда множитель.
MAX_CASE_ELEMS = 8 << 20          # 8M элементов = 32 МБ fp32 на копию
EMB_ROWS = 2048                   # срез больших таблиц, а не вся vocab


def swapusage():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout.strip()
    return out


def digest(t: torch.Tensor) -> str:
    """sha256 сырых байт + форма + тип. Бит-в-бит, без хранения тензора."""
    a = t.detach().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(a.shape)).encode())
    h.update(str(a.dtype).encode())
    h.update(a.cpu().numpy().tobytes())
    return h.hexdigest()


def cases(sd):
    """(имя, тензор) — смесь настоящих весов и синтетики, потоково.

    Настоящие: по представителю каждой группы, включая ragged-случай
    att.w1 (число блоков = ceil(IN/gs), а не //). emb/head берутся
    СРЕЗОМ в EMB_ROWS строк: раскладка блочная вдоль входной оси, то
    есть строки независимы, и срез покрывает ровно ту же математику при
    ограниченной памяти. Полная таблица в гейте не нужна и однажды уже
    стоила паники.
    """
    want = [("proj", "blocks.3.att.key.weight", None),
            ("cmix_k", "blocks.3.ffn.key.weight", None),
            ("cmix_v", "blocks.3.ffn.value.weight", None),
            ("emb", "emb.weight", EMB_ROWS),
            ("head", "head.weight", EMB_ROWS),
            ("w1_ragged", "blocks.3.att.w1", None),
            ("g1", "blocks.3.att.g1", None)]
    for tag, key, rows in want:
        if key not in sd or sd[key].dim() != 2:
            continue
        w = sd[key]
        if rows is not None:
            w = w[:rows]
        w = w.float().contiguous()
        if w.numel() > MAX_CASE_ELEMS:
            raise RuntimeError(
                f"{key}: {w.numel()} элементов > лимита {MAX_CASE_ELEMS}. "
                f"Ограничьте срез — гейт не место для полноразмерных таблиц.")
        yield f"real::{tag}::{key}", w
    g = torch.Generator().manual_seed(20260810)
    yield "synth::gauss", torch.randn(128, 512, generator=g)
    z = torch.randn(64, 256, generator=g)
    z[7] = 0.0            # строка-константа: half-underflow у scale
    z[9] = 1e-9
    yield "synth::degenerate", z


def iter_results(fns, sd):
    """Генератор (имя, тензор). ОДИН выход жив за раз — вызывающая
    сторона обязана его отпустить (хеш посчитан, ссылка не держится)."""
    gw, sym, mx4 = fns
    g = torch.Generator().manual_seed(7)
    for name, w in cases(sd):
        ex2 = torch.rand(w.shape[-1], generator=g) + 0.05
        for bits in (4, 5, 6):
            yield f"{name}|asym_sb6|{bits}", gw(w, bits, 32, sb=8, sb_bits=6)
            yield f"{name}|asym_sb6_search|{bits}", gw(w, bits, 32, sb=8, sb_bits=-6)
            yield f"{name}|asym_sb6_aw|{bits}", gw(w, bits, 32, sb=8, sb_bits=-6, ex2=ex2)
            parts = gw(w, bits, 32, sb=8, sb_bits=-6, ex2=ex2, return_parts=True)
            for k in ("q", "deq", "scale", "mn", "qs", "qm", "d", "dm"):
                if k in parts:
                    yield f"{name}|parts_aw|{bits}|{k}", parts.pop(k)
            del parts
        yield f"{name}|asym_plain|6", gw(w, 6, 64)
        for bits in (5, 6):
            yield f"{name}|sym|{bits}", sym(w, bits, gs=16, sb=16, ex2=None, search=True)
            yield f"{name}|sym_plain|{bits}", sym(w, bits, gs=16, sb=16, ex2=None, search=False)
            yield f"{name}|sym_aw|{bits}", sym(w, bits, gs=16, sb=16, ex2=ex2, search=True)
            yield f"{name}|sym_spqr|{bits}", sym(w, bits, gs=16, sb=16, search=True,
                                                outlier_frac=0.01)
        yield f"{name}|mxfp4", mx4(w, 32)
        yield f"{name}|mxfp4_spqr", mx4(w, 32, 0.01)


def load_sd():
    return torch.load(CKPT, map_location="cpu", mmap=True)


def hashes(fns, sd, progress=False):
    """{имя: sha256}. Пик по памяти = один выход + воркспейс одной функции."""
    out = {}
    for i, (name, t) in enumerate(iter_results(fns, sd)):
        out[name] = digest(t)
        del t
        if progress and (i + 1) % 50 == 0:
            print(f"  {i+1} ...", flush=True)
    return out


def main():
    from rwkv_quant.formats import writer
    print(f"своп до:    {swapusage()}")
    fns = (writer._groupwise_fake_dequant,
           writer._groupwise_sym_fake_dequant,
           writer._mxfp4_fake_dequant)
    h = hashes(fns, load_sd(), progress=True)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w") as f:
        json.dump({"ckpt": os.path.basename(CKPT), "emb_rows": EMB_ROWS,
                   "hashes": h}, f, indent=1, sort_keys=True)
    print(f"эталон: {len(h)} случаев -> {GOLDEN} "
          f"({os.path.getsize(GOLDEN)/1024:.0f} КБ)")
    print(f"своп после: {swapusage()}")


if __name__ == "__main__":
    main()
