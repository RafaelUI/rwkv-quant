"""
Детект схемы именования тензоров в чекпоинте RWKV-7.

Существует минимум два формата на практике:
  - "custom": кастомные обучающие пайплайны (напр. тренировка своей модели
    с нуля) именуют тензоры как blocks.N.tmix.r_proj / w_lora_A / w_lora_B(+bias)...
  - "world": официальные чекпоинты BlinkDL (rwkv7-*-world, G1/G1H и т.д.)
    именуют как blocks.N.att.receptance / w0,w1,w2 / a0,a1,a2...

Математика forward pass идентична для обеих схем (см. rwkv7_ref.py) --
разница только в том, как теги весов в state_dict раскладываются по
внутреннему представлению TMix/CMix.
"""


def detect_naming(ckpt_path: str, state_dict) -> str:
    if ckpt_path.endswith(".pth"):
        return "world"
    return "custom"
