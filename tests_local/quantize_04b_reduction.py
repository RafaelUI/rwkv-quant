import copy
from rwkv_quant import quantize
from rwkv_quant.presets import REDUCTION

cfg = copy.deepcopy(REDUCTION)
cfg.act_stats_path = "/tmp/act_stats_04b.pt"

quantize(
    "/tmp/ckpt_step135000_world.pth",
    "/tmp/rwkv7-0.4b-embed-reduction.rwkvq",
    config=cfg,
    real_gw=True,
    verbose=True,
)
