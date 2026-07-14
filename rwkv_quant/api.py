"""
Высокоуровневый API для быстрого старта. Для точного контроля над
per-group битностью, outlier-долей и clip-percentile используйте
rwkv_quant.calibration.QuantConfig напрямую (см. calibrate() ниже
и examples/quantize_ru60m.md).
"""
from .presets import PRESETS
from .calibration.group_config import QuantConfig


def quantize(checkpoint_path: str, output_path: str, preset: str = "reduction",
             config: "QuantConfig | None" = None):
    """
    Quick-start: quantize(ckpt, out, preset="compression")
    Advanced:    quantize(ckpt, out, config=QuantConfig(proj=4, ...))

    preset игнорируется, если передан config.
    """
    if config is None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}, choose from {list(PRESETS)}")
        config = PRESETS[preset]
    # TODO: models.rwkv7_ref.RWKV7Ref(checkpoint_path) -> apply config ->
    # formats.writer.save(output_path)
    raise NotImplementedError("wire up models/formats -- next step")


def calibrate(checkpoint_path: str, eval_corpus_path: str) -> "QuantConfig":
    """Прогоняет outlier_scan + ablation на конкретном чекпоинте и
    возвращает рекомендованный QuantConfig вместо пресета "с потолка"."""
    raise NotImplementedError("wire up calibration/ pipeline -- next step")
