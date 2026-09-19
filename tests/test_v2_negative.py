"""Негативный контроль бит-точности: тест test_v2_format мог бы проходить
'тривиально' (общий кэш, пронос dense). Здесь: (1) порча одного байта
каждого packed-поля ДОЛЖНА менять деквант; (2) деквант != исходный w."""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from rwkv_quant.formats.writer import quantize_tensor
from rwkv_quant.formats.reader import _dequantize_one
from rwkv_quant.formats import codec
from test_v2_format import CHAMPION

CKPT = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
sd = torch.load(CKPT, map_location="cpu")

for key in ["blocks.0.att.receptance.weight", "blocks.0.ffn.key.weight",
            "blocks.0.att.w1"]:
    w = sd[key]
    real = quantize_tensor(key, w, CHAMPION, real_gw=True)
    base = _dequantize_one(real)
    # Сравнение -- в ориентации nn.Linear [out, in] с обеих сторон. Сырые
    # LoRA-ключи world-чекпоинта лежат в [in, out] (codec.is_raw_lora_world);
    # с 28.08 writer квантует их ПОСЛЕ транспонирования и ставит
    # transposed=False, то есть деквант уже [out, in]. Флаг transposed=True
    # значил бы «хранится сыро, транспонируй после декванта» (файлы до 28.08,
    # RWKVQ_LORA_TRANSPOSE=0). Прежняя редакция сравнивала деквант с СЫРЫМ w
    # и падала на несовпадении форм 96 против 2048 -- ошибка теста.
    wref = w.T if codec.is_raw_lora_world(key) else w
    lin = base.T if real.transposed else base
    assert lin.shape == wref.shape, (
        f"{key}: форма декванта {tuple(lin.shape)} не совпала с исходной "
        f"{tuple(wref.shape)} (transposed={real.transposed})")
    dw = (lin.float() - wref.float()).abs().max().item()
    assert dw > 0, f"{key}: деквант побитово равен исходному w -- квантования не было!"
    fields = [f for f in ("codes_packed", "codes", "gw_qsqm", "gw_qh")
              if getattr(real, f) is not None]
    for f in fields:
        r2 = copy.deepcopy(real)
        t = getattr(r2, f)
        flat = t.view(-1)
        flat[len(flat)//2] ^= 0xFF if t.dtype == torch.uint8 else 0x55
        deq2 = _dequantize_one(r2)
        changed = not torch.equal(base, deq2)
        print(f"{key:38s} corrupt {f:12s} -> deq changed: {changed}  "
              f"(|deq-w|max={dw:.4f})")
        assert changed, f"{key}: порча {f} не изменила деквант -- поле мёртвое!"
print("NEGATIVE CONTROLS PASSED")
