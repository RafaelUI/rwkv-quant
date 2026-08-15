"""Гейт: нарезка полосами в симметричной ветке бит-в-бит.

Асимметричная ветка режется полосами с 12.08 и покрыта
test_groupwise_parity; симметричная нарезки не имела вовсе, из-за чего
была неприменима к emb/head на 2.9B (168M элементов x полтора десятка
копий грид-поиска). Этот гейт требует РАВЕНСТВА, а не близости: полосы --
это перестановка порядка вычислений, а не другая арифметика, и любое
расхождение здесь означает, что какая-то величина всё-таки смотрит на
соседние строки.

Формы берутся из НАСТОЯЩЕГО чекпоинта, а не выдумываются: синтетика,
которую нельзя предъявить в реальном файле, покрытием не является
(закон 17). Обязательно есть случай с IN, не кратным gs*sb -- у
`blocks.N.att.w1` при [2048, 96] и gs=16, sb=16 паддинг ненулевой, и
именно там нарезка могла бы разъехаться.

    python tests/test_sym_chunk_parity.py [ckpt] [device]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwkv_quant.calibration import groupwise as gw

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
DEV = sys.argv[2] if len(sys.argv) > 2 else "mps"


def main():
    sd = torch.load(CKPT, map_location="cpu", mmap=True)
    # по одному представителю каждой формы, плюс ragged-случай
    want = ["emb.weight", "head.weight", "blocks.3.ffn.key.weight",
            "blocks.3.att.key.weight", "blocks.3.att.w1"]
    keys = [k for k in want if k in sd]
    if not keys:
        raise SystemExit(f"в {CKPT} нет ни одного из {want}")

    fails = 0
    for key in keys:
        w = sd[key]
        if w.dim() != 2:
            continue
        # emb/head целиком в fp32 на MPS -- это и есть та память, от
        # которой мы уходим; для гейта хватает среза строк, форма вдоль
        # ВХОДНОЙ оси при этом сохраняется полностью
        if w.shape[0] > 8192:
            w = w[:8192]
        w = w.to(DEV).to(torch.bfloat16)
        ex2 = torch.rand(w.shape[1], device=DEV) + 0.5
        for bits in (6,):
            for search in (True, False):
                for e in (None, ex2):
                    for frac in (0.0, 0.01):
                        # одна полоса: порог заведомо больше тензора
                        gw._CHUNK_BYTES = 1 << 40
                        a = gw.groupwise_sym_fake_dequant(
                            w, bits, gs=16, sb=16, ex2=e, search=search,
                            outlier_frac=frac)
                        # много полос: 1 МБ -> у [8192, 2048] это 32 полосы
                        gw._CHUNK_BYTES = 1 << 20
                        b = gw.groupwise_sym_fake_dequant(
                            w, bits, gs=16, sb=16, ex2=e, search=search,
                            outlier_frac=frac)
                        ok = torch.equal(a, b)
                        if not ok:
                            fails += 1
                            d = float((a.float() - b.float()).abs().max())
                            print(f"РАСХОЖДЕНИЕ {key} bits={bits} "
                                  f"search={search} aw={e is not None} "
                                  f"frac={frac}: max|Δ|={d:.3e}")
                        del a, b
        n_chunks = max(1, (w.shape[0] * w.shape[1] * 4) // (1 << 20))
        print(f"{key:32s} {tuple(w.shape)} -> 8 случаев, ~{n_chunks} полос: "
              f"{'ok' if not fails else 'см. выше'}")
    gw._CHUNK_BYTES = int(os.environ.get("RWKVQ_GW_CHUNK_MB", "128")) << 20
    print("\nЗЕЛЁНЫЙ" if not fails else f"\nКРАСНЫЙ: {fails} расхождений")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
