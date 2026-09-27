"""Гейт (28.09): точно нулевой суперблок в asym_sb6 не даёт NaN.

До правки _gw_one: все min суперблока == 0 и dm в half-underflow -> qm = 0/0 = NaN,
деквант суперблока NaN (fake-путь), в файл -- NaN, приведённый к int8. Всплыло на GPTQ:
он обнуляет почти мёртвые строки g1j 1.5B (blocks.4.ffn.key) точно. Проверяется: для
bits 4/5/6, с поиском и без, с ex2 и без -- выход конечен, нулевой суперблок и нулевая
строка деквантуются в 0, строка с одним нулевым суперблоком вне его совпадает с той же
строкой, где суперблок не нулевой, НЕ хуже чем побитно для остальных суперблоков (строки
независимы: считаем строку отдельно). Побитность на реальных тензорах против кода ДО правки
-- tests/_sess/gw_zero_sb_freeze_2809.py (125 записей, 0 расхождений).
Мутация: без nan_to_num в _gw_one гейт падает (проверено 28.09)."""
import sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rwkv_quant.calibration import groupwise as gw
from rwkv_quant.formats import writer as W


def main():
    torch.manual_seed(zlib.crc32(b"zero_sb"))
    w = torch.randn(4, 2048) * 0.02
    w[1, 256:512] = 0.0          # нулевой суперблок посреди строки
    w[2] = 0.0                    # нулевая строка
    w[3, :256] = 1e-9             # константа-крошка (не ноль: прежде тоже конечна)
    g = torch.Generator().manual_seed(zlib.crc32(b"ex2"))
    ex = torch.rand(2048, generator=g) + 0.1
    n = 0
    for bits in (4, 5, 6):
        for sbb in (-6, 6):
            for e in (ex, None):
                o = gw._gw_one(w.clone(), bits, 32, 8, sbb, e, False)
                assert torch.isfinite(o).all(), (bits, sbb, e is None, (~torch.isfinite(o)).sum(1).tolist())
                assert (o[1, 256:512] == 0).all() and (o[2] == 0).all(), (bits, sbb)
                # вне нулевого суперблока строка 1 квантуется независимо от него:
                # те же суперблоки в отдельной строке без нулевого -- побитно
                alone = gw._gw_one(torch.cat([w[1, :256], w[1, 512:]]).view(1, -1).clone()[:, :1792], bits, 32, 8, sbb,
                                   torch.cat([e[:256], e[512:]])[:1792] if e is not None else None, False)
                assert torch.equal(o[1, :256], alone[0, :256]), (bits, sbb, "суперблок 0")
                n += 1
        qt = W._make_qt_gw_sb6("syn", "cmix", bits, w, 32, ex, search=True)
        for k, v in vars(qt).items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                assert torch.isfinite(v.float()).all(), (bits, k)
    print("test_gw_zero_superblock: %d комбинаций, ЗЕЛЁНЫЙ" % n)


if __name__ == "__main__":
    main()
