"""
Формат .rwkvq.

`codec` импортируется сразу -- он на numpy и torch не требует. Всё
остальное лениво: см. докстринг rwkv_quant/__init__.py, там объяснено,
почему это обязательное свойство, а не оптимизация.
"""
import importlib

from . import codec  # noqa: F401  -- numpy-only, безопасно на любом окружении

_LAZY = {
    "QuantizedTensor": ".schema",
    "QuantizedCheckpoint": ".schema",
    "save": ".writer",
    "save_rwkvq": ".writer",
    "quantize_file": ".writer",
    "quantize_tensor": ".writer",
    "config_to_json": ".writer",
    "load_raw": ".reader",
    "load_dequantized": ".reader",
}

__all__ = list(_LAZY) + ["codec"]


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"модуль {__name__} не имеет атрибута {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
