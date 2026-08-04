"""
Цена загрузки .rwkvq: прежний pickle против нового safetensors.

ОДИН КОНФИГ НА ПРОЦЕСС (законы 11 и 13): память меряется по свободной
доле системы, а не по RSS — для mmap RSS врёт, страницы файла считаются
резидентными, хотя они чистые и вытесняются без свопа. Чередование
между процессами делает вызывающий скрипт.

    python tests/bench_rwkvq_load.py <model.rwkvq>

Печатает: контейнер, время загрузки, минимум свободной памяти за
загрузку, дельту свопа, maxrss (справочно, см. закон 11).
"""
import os
import resource
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rwkv_quant.formats import codec  # noqa: E402
from rwkv_quant.formats.reader import load_raw  # noqa: E402


def _free_pct():
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return -1


def _swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    return float(out.split("used =")[1].split("M")[0])


def main():
    path = sys.argv[1]
    kind = ("pickle" if open(path, "rb").read(4) == codec.MAGIC_ZIP
            else "safetensors")

    stop, lo = threading.Event(), [_free_pct()]

    def sampler():
        while not stop.is_set():
            lo[0] = min(lo[0], _free_pct())
            time.sleep(0.05)

    th = threading.Thread(target=sampler, daemon=True)
    swap0 = _swap_mb()
    th.start()
    t0 = time.perf_counter()
    ckpt = load_raw(path)
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    swap1 = _swap_mb()

    n_buf = sum(1 for qt in ckpt.tensors.values()
                for f in ("codes", "codes_packed", "scale", "dense", "gw_d",
                          "gw_dm", "gw_qsqm", "gw_qh", "gw_qh2", "gw_scale",
                          "gw_min")
                if getattr(qt, f) is not None)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

    print(f"{kind:<12} {os.path.getsize(path) / 1e6:7.1f} МБ  "
          f"{len(ckpt.tensors)} тензоров / {n_buf} буферов  "
          f"загрузка {dt:6.2f} с  мин. свободно {lo[0]}%  "
          f"своп {swap1 - swap0:+.0f} МБ  maxrss {rss:.2f} ГБ")


if __name__ == "__main__":
    main()
