"""ГЕЙТ: файлы данных доезжают в СОБРАННЫЙ пакет, а не только лежат в дереве.

Написан после 04.09, когда выяснилось, что колесо не содержало ни одного
файла данных: каталог `rwkv_quant/data/` пакетом не является, поэтому
`packages.find` его не брал, а `package-data` и `MANIFEST.in` в проекте
отсутствовали. При этом всё работало у любого, кто запускает из дерева
исходников: `DATA_DIR` считается от `__file__`, и файл лежал рядом. То
есть отказ был виден ТОЛЬКО снаружи дерева -- у того, кто поставил пакет.

Цена молчания измерена: умолчательный путь `quantize(act_stats="auto")`
читает `rwkv_quant/data/calib_corpus.txt`, значит публичный API падал у
каждого установившего пакет.

Проверяются оба канала поставки, потому что механизмы у них разные и
почин одного не доказывает починку второго:

  1. Колесо (`pip wheel`) содержит КАЖДЫЙ файл из `rwkv_quant/data/`.
  2. Содержимое совпадает с исходником побайтово, а не только по имени.
  3. sdist (`python -m build --sdist`) содержит их же.

И отдельно: если файлов данных в дереве не нашлось вовсе, гейт КРАСНЕЕТ,
а не отчитывается зелёным, ничего не проверив (закон 31).

СОБИРАТЬ ТОЛЬКО ИЗ ЧИСТОЙ КОПИИ ДЕРЕВА. Первая редакция этого гейта
собирала прямо в рабочем каталоге и на мутации (возврат прежнего
pyproject.toml) осталась ЗЕЛЁНОЙ: build/lib и egg-info/SOURCES.txt
переживают правку и подкладывают файлы, положенные туда предыдущей
удачной сборкой. То есть гейт мерил остатки инструмента, а не упаковку --
и был бы зелёным ровно в том случае, ради которого написан.

    python tests/test_packaging_data.py
"""
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "rwkv_quant", "data")
ARCNAME = "rwkv_quant/data/%s"


def _sources():
    if not os.path.isdir(DATA):
        return []
    return sorted(f for f in os.listdir(DATA)
                  if not f.startswith(".") and
                  os.path.isfile(os.path.join(DATA, f)))


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-42s %s %s" % (name, "ок" if cond else "ПРОВАЛ", detail))

    files = _sources()
    # ОТКАЗ ПРИ НУЛЕ СЛУЧАЕВ: гейт, которому нечего проверять, обязан
    # краснеть. Иначе он зелёный ровно в тот день, когда каталог данных
    # переименовали и он перестал попадать куда бы то ни было.
    check("файлы данных в дереве найдены", len(files) > 0,
          "%s" % (files or DATA))
    if not files:
        print("\n[FAIL] упаковка данных")
        return False

    with tempfile.TemporaryDirectory() as td:
        # Чистая копия: см. докстринг. Без неё мутация не краснеет.
        srcdir = os.path.join(td, "src")
        shutil.copytree(ROOT, srcdir, ignore=shutil.ignore_patterns(
            ".git", "build", "dist", "*.egg-info", "__pycache__", "*.pyc"))
        r = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", srcdir, "--no-deps",
             "--no-build-isolation", "--no-cache-dir", "-q", "-w", td],
            capture_output=True, text=True)
        check("колесо собралось", r.returncode == 0, r.stderr.strip()[-160:])
        whls = [f for f in os.listdir(td) if f.endswith(".whl")]
        check("колесо на месте", len(whls) == 1, str(whls))
        if whls:
            with zipfile.ZipFile(os.path.join(td, whls[0])) as z:
                names = set(z.namelist())
                for f in files:
                    arc = ARCNAME % f
                    here = arc in names
                    check("в колесе: %s" % f, here)
                    if here:
                        want = open(os.path.join(DATA, f), "rb").read()
                        got = z.read(arc)
                        check("  байт-в-байт: %s" % f, got == want,
                              "%d байт" % len(got))

        # sdist -- ВТОРОЙ канал поставки, механизм включения у него свой.
        r2 = subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--no-isolation",
             "-o", td, srcdir], capture_output=True, text=True)
        if r2.returncode != 0:
            check("sdist собрался", False, r2.stderr.strip()[-160:])
        else:
            tars = [f for f in os.listdir(td) if f.endswith(".tar.gz")]
            check("sdist на месте", len(tars) == 1, str(tars))
            if tars:
                with tarfile.open(os.path.join(td, tars[0])) as t:
                    inner = set("/".join(n.split("/")[1:])
                                for n in t.getnames())
                    for f in files:
                        check("в sdist: %s" % f, (ARCNAME % f) in inner)

    print("\n[%s] упаковка данных" % ("OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
