"""Собрать .rwkvq для конфига из ablate_sym_composite (артефакт для
замеров префилла и для выкатки). Один конфиг на процесс."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwkv_quant.formats.writer import quantize_file  # noqa: E402
import ablate_sym_composite as comp  # noqa: E402

name = sys.argv[1]
out = sys.argv[2]
quantize_file(comp.CKPT, out, comp.CONFIGS[name](), real_gw=True)
