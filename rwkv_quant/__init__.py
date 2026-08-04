"""
rwkv_quant: квантование RWKV-7 и формат .rwkvq.

ИМПОРТЫ ЛЕНИВЫЕ, и это не стиль, а требование. `formats/codec.py` --
нормативная torch-free читалка формата, ради которой всё и делалось:
её потребители (rwkv-metal на MLX, SwiftRWKV) torch не имеют и иметь не
должны. Пока этот файл делал `from .api import quantize`, любой импорт
внутри пакета тянул torch, и `from rwkv_quant.formats import codec`
падал с ModuleNotFoundError. То есть torch-free путь был ЗАЯВЛЕН, но
недостижим -- поймано только когда его попробовали импортировать в
окружении без torch.

Публичный API не изменился: `rwkv_quant.quantize(...)` работает как
раньше, просто torch подтягивается в момент обращения, а не импорта.
Гейт: tests/test_torch_free_import.py.
"""
import importlib

_LAZY = {
    "quantize": ".api",
    "calibrate": ".api",
    "QuantConfig": ".calibration",
    "GROUPS": ".calibration",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        value = getattr(mod, name)
        globals()[name] = value          # второй раз уже без importlib
        return value
    raise AttributeError(f"модуль {__name__} не имеет атрибута {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
