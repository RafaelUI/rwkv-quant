"""Снимает эталон для tests/test_group_split_compat.py.

Запускать на дереве ДО разделения emb_head:
    git stash && python tests/make_group_split_golden.py && git stash pop
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_group_split_compat as t


def main():
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True).stdout.strip()
    print(f"своп до: {sw}")
    data = t.collect()
    os.makedirs(os.path.dirname(t.GOLDEN), exist_ok=True)
    with open(t.GOLDEN, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print(f"эталон: {len(data['hashes'])} буферов, "
          f"{len(data['reprs'])} пресетов -> {t.GOLDEN} "
          f"({os.path.getsize(t.GOLDEN)/1024:.0f} КБ)")
    for k, v in data["reprs"].items():
        print(f"  {k}: {v}")
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True).stdout.strip()
    print(f"своп после: {sw}")


if __name__ == "__main__":
    main()
