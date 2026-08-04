"""
Гейт: codec читается БЕЗ torch, и это проверяется, а не декларируется.

Зачем отдельный гейт. Весь смысл codec.py -- дать rwkv-metal и
SwiftRWKV читалку формата без torch. Пока `rwkv_quant/__init__.py`
делал `from .api import quantize`, а `formats/__init__.py` --
`from .writer import save`, ЛЮБОЙ импорт внутри пакета тянул torch, и
`from rwkv_quant.formats import codec` падал с ModuleNotFoundError на
машине потребителя. Заявление о torch-free жило полдня и было ложным,
потому что его никто не пробовал выполнить в окружении без torch.

Здесь torch блокируется на уровне механизма импорта, поэтому гейт
осмыслен и там, где torch установлен, -- а это как раз машина
разработчика, где иначе ничего бы не поймалось.

    python tests/test_torch_free_import.py
"""
import os
import subprocess
import sys
import textwrap

REPO = os.path.join(os.path.dirname(__file__), "..")

# Дочерний процесс: torch запрещён к импорту, дальше делаем ровно то,
# что делает потребитель формата. Отдельный процесс обязателен -- в
# текущем torch уже импортирован тестами и блокировка ничего не значила бы.
CHILD = textwrap.dedent("""
    import sys

    class BlockTorch:
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch заблокирован гейтом")
            return None

    sys.meta_path.insert(0, BlockTorch())
    sys.path.insert(0, {repo!r})

    from rwkv_quant.formats import codec
    import numpy as np

    # не просто импорт: прогоняем настоящую работу формата
    q = np.arange(64, dtype=np.uint8).reshape(2, 32) % 16
    p = codec.pack_nib_block(q.astype(np.uint8), 32)
    back = codec.unpack_nib_block(p, 32)
    assert (back == q).all(), "распаковка не сошлась"

    w = codec.pack_mlx_affine(q.reshape(2, 1, 32), 6)
    assert codec.unpack_mlx_affine(w, 6, 1).reshape(2, 32).tolist() == q.tolist()

    assert codec.FORMAT == "rwkvq"
    assert "torch" not in sys.modules, "torch всё-таки подтянулся"
    print("OK", codec.FORMAT_VERSION)
""").format(repo=os.path.abspath(REPO))


def main():
    print("импорт codec с заблокированным torch")
    r = subprocess.run([sys.executable, "-c", CHILD],
                       capture_output=True, text=True)
    ok = r.returncode == 0 and r.stdout.startswith("OK")
    print(f"  {'ok  ' if ok else 'FAIL'} дочерний процесс: "
          f"rc={r.returncode} {r.stdout.strip()}")
    if not ok:
        print(r.stderr.strip()[-1500:])

    # и контроль в другую сторону: обычный путь через torch не сломан
    r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {os.path.abspath(REPO)!r}); "
         "import rwkv_quant; "
         "assert callable(rwkv_quant.quantize); "
         "from rwkv_quant.formats import load_raw, save_rwkvq; "
         "print('OK')"],
        capture_output=True, text=True)
    ok2 = r2.returncode == 0 and r2.stdout.strip() == "OK"
    print(f"  {'ok  ' if ok2 else 'FAIL'} ленивый API не сломан: "
          f"rc={r2.returncode} {r2.stdout.strip()}")
    if not ok2:
        print(r2.stderr.strip()[-1500:])

    print("\nГЕЙТ " + ("ПРОЙДЕН" if ok and ok2 else "ПРОВАЛЕН"))
    return 0 if ok and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
