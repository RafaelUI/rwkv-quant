# rwkv-quant

Quantization toolkit for RWKV-7, built around a finding that surprised us:
**quantization sensitivity does not transfer across model scale**, and the reason
is per-channel outlier values in specific RWKV-7 components (`r_k`, `k_k`, `k_a`,
the LoRA-style decay/gate projections). This repo packages the calibration,
outlier handling, and portable quantized format needed to compress RWKV-7
checkpoints without hand-tuning per model.

## Quick start

```python
from rwkv_quant import quantize

# ~2x compression, near-lossless (+1.6% ppl on our 1.5B reference run)
quantize("model.pth", "model.rwkvq", preset="reduction")

# ~3.5x compression, moderate quality cost (+37.6% ppl on our reference run)
quantize("model.pth", "model.rwkvq", preset="compression")
```

Presets are a starting point, calibrated on `rwkv7-g1h-1.5b-ctx10240` — see
[Why presets aren't universal](#why-quantization-sensitivity-doesnt-transfer-across-scale)
below. For a checkpoint-specific config:

```python
from rwkv_quant import quantize, calibrate

config = calibrate("model.pth", "eval_corpus.pt")   # runs outlier scan + ablation
quantize("model.pth", "model.rwkvq", config=config)
```

Or build a config by hand for full control over every group:

```python
from rwkv_quant.calibration import QuantConfig

config = QuantConfig(
    proj=4, cmix=4, emb_head=4,           # dense matrices
    w_lora=4, a_lora=4, v_lora=4, g_lora=4,  # LoRA decay/ICL-rate/value/gate
    small=6,                               # k_k, k_a, r_k
    outlier_fracs={"proj": 0.02, "cmix": 0.02, "emb_head": 0.02},
    clip_percentiles={"small": 99.9},
)
```

## Why quantization sensitivity doesn't transfer across scale

Ran weight-only fake-quantization ablations on two RWKV-7 checkpoints — a
custom 61M Russian model (18L/D448) and BlinkDL's official 1.5B G1H — across
8 parameter groups: `proj` (R/K/V/O), the four LoRA-style projections
(`w_lora`/`a_lora`/`v_lora`/`g_lora` — decay, in-context learning rate, value
residual, output gate), `small` (k_k/k_a/r_k), `cmix` (FFN), `emb_head`.

**On 61M**, every LoRA-ish component survived INT2 with <2% ppl loss. Only the
full-rank `proj`/`cmix` matrices were fragile at INT2.

**On 1.5B**, that pattern inverts. At INT4 alone:

```
group      61M Δppl    1.5B Δppl
proj        +0.55%      +19.97%
cmix        +1.41%      +48.09%
emb_head    +4.03%      +97.17%
g_lora      -0.02%       +7.55%
small       +0.05%   +21,784,766%   <- yes, really
w/a/v_lora  ~0.02%       <0.5%
```

`small` and `g_lora` go from "quantize freely" to "catastrophic" as the model
scales up. **Assuming a group is safe because it was safe on a smaller
checkpoint is a real trap** — this repo's `calibrate()` exists specifically so
you don't have to guess.

### Root cause

Per-row max/mean ratios of 40–96x show up in `r_k`, `k_k`, `k_a`, and even in
`proj`/`cmix`. With symmetric per-channel quantization, scale = max/qmax — a
single 96x outlier in an otherwise tight channel forces the other ~63 "normal"
values into 1–2 quantization codes, destroying the channel. Same mechanism as
`LLM.int8()`'s outlier features in transformers, showing up in RWKV-7's
LoRA-style decay/gate projections instead of attention. Confirmed the failure
is systemic, not corpus noise: at INT4, all held-out eval chunks blew up
simultaneously, not a localized artifact.

### Two mitigations, and they are not interchangeable

**Percentile clipping** (scale from the 99th percentile instead of abs-max)
fully rescued `small`:

```
small @ INT6, no clip:        +11.55% ppl
small @ INT6, clip p99.9:      +1.60% ppl
small @ INT5, clip p99.0:      +3.94% ppl  (better than unclipped INT6!)
```

But it **hurt** `proj`/`cmix` — made INT4 slightly worse. The dense matrices'
outlier tail is trained signal, not noise; clipping throws away information.

**SpQR-style sparse outlier extraction** (keep the top-k% largest-magnitude
values per row exact in bf16, quantize the rest with a clean scale) worked
much better for `proj`/`cmix`/`emb_head`:

```
group      INT4 no fix   INT4 + clip     INT4 + SpQR(1%)
proj         +19.97%      +22.64% (worse)   +5.62%
cmix         +48.09%      +51.63% (worse)  +13.00%
emb_head     +97.17%      +14.22%            +6.87%
```

Sweeping outlier fraction shows diminishing returns past ~2%:

```
outlier%   Δppl      compression
1%        +54.86%       3.74x
2%        +37.61%       3.52x
3%        +32.68%       3.32x
4%        +31.51%       3.14x
```

## Caveats

Evaluated on a small held-out text corpus and fake quantization only (no
packed kernels / measured speedup yet — that's what `backends/` is for).
Presets above are calibrated on one 1.5B checkpoint; recalibrate for other
sizes. `g_lora`'s catastrophic INT2 failure hasn't been tested with SpQR yet.

## Repo layout

See [STRUCTURE.md](./STRUCTURE.md).

## Backends

Calibration (`rwkv_quant.calibration`) is pure PyTorch and platform-agnostic.
Actual quantized inference is platform-specific:
- `backends/metal/` — built on top of [rwkv-metal](https://github.com/) kernels
- `backends/cuda/` — CUDA path

Both consume the same portable `.rwkvq` format from `rwkv_quant.formats`.
