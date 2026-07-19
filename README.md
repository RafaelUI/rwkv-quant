# rwkv-quant

Quantization toolkit **and Metal inference backend** for RWKV-7 on Apple
Silicon. Portable `.rwkvq` checkpoint format, outlier-aware calibration, and
custom Metal GEMV kernels that decode the quantized format directly — no
dequantized weight copy in memory.

Reference model: `rwkv7-g1h-1.5b` (BlinkDL G1H 1.5B, bf16 2953 MB).
Reference machine: M4 MacBook Air 16 GB (base chip, fanless).

| preset | size | Δppl vs bf16 | decode | prefill T=1024 |
|---|---|---|---|---|
| bf16 reference | 2953 MB | — | — | — |
| `reduction` (all-INT6 group-wise) | 1256 MB (2.35x) | **+0.12 %** | 17.7 ms/tok | 437 t/s |
| `compression` (INT4/INT5 mixed) | 971 MB (3.04x) | +2.47 % | **14.8 ms/tok** (14.0 with n-gram speculation) | 545 t/s |

For scale: the community MLX affine-INT6 build of the same model is the same
size as `reduction` (1272 MB) and runs 15.1 ms/tok, but costs +1.06 % ppl —
roughly 9x the quality loss of `reduction`. Our `compression` preset is both
faster than it and 300 MB smaller. Speed was closed by kernel work while
keeping the native format (see [Kernels](#kernels)).

## Quick start

```python
from rwkv_quant import quantize

# near-lossless, 2.35x smaller than bf16 (+0.12% ppl on the 1.5B reference)
quantize("model.pth", "model.rwkvq", preset="reduction")

# 3.04x smaller, moderate cost (+2.47% ppl), fastest decode
quantize("model.pth", "model.rwkvq", preset="compression")
```

Both presets use activation-weighted (AW) scale search and expect activation
statistics at the path set in `QuantConfig.act_stats_path`
(`tests/collect_act_stats.py` produces them in ~30 s).

Inference (Metal / MLX):

```python
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("model.rwkvq"))
state = model.init_state(1)
logits, state = model.forward_stateful(token_ids, state)   # prefill / decode
```

Presets are calibrated on `rwkv7-g1h-1.5b` — see
[Why presets aren't universal](#why-quantization-sensitivity-doesnt-transfer-across-scale).
For a checkpoint-specific config run `calibrate()` or build a `QuantConfig`
by hand (per-group bits, group scale sizes, scale modes, clipping).

## Format

`.rwkvq` stores group-wise asymmetric quantization ("sb6"): blocks of 32
weights share a 6-bit scale/min pair (`qs`/`qm`), superblocks of 256 share an
fp16 pair (`d`/`dm`) that the 6-bit pairs multiply. Codes are packed as
nibbles; INT5/INT6 add one/two bit-planes on top. Scale search is
activation-weighted where it helps (per-group setting). The format is
backend-independent; per-tensor bits and modes live in the file, not in code.

A finding that shaped the presets: **granularity beats bits.** Group-wise
sb6 at INT4/5 replaced an earlier per-row + SpQR-outlier scheme of the same
size with roughly half the quality loss. Sub-nibble packing (sub-887 MB at
sane ppl) does not fit the nibble container — that's a future format, not a
tuning exercise.

## Kernels

`backends/metal/` decodes sb6 on the fly inside GEMV — weights never exist
dequantized in memory. Highlights (all validated bit-exact against the
reference implementation, so quality numbers carry over without re-eval):

- **Layout borrowed from MLX `qmv`** (PR #1503): N simdgroups x R rows per
  threadgroup, dispatch table per matrix shape.
- **Interleaved load-time repack**: codes + bit-planes contiguous per block,
  quality scalars as `uchar2`/`half2` — 4-5 memory transactions per
  (row, block) instead of 7. On-disk format untouched; memory stays 1x.
- **Bit-plane decode via multiply trick** (`(nib * 0x00204081) & 0x01010101`)
  — ~3x less ALU per plane; this is what unlocked INT6 decode speed
  (head INT6: 88 → 103 GB/s, ~85 % of the M4's DRAM bandwidth).
- **Batched verify kernels** (weights decoded once per N columns) for
  speculative decoding; n-gram prompt-lookup speculation ships in the demo
  scripts (1.08-1.25x on repetitive text, never slower).
- Fused r/k/v projection launch and fused lerp/LoRA batching in the decode
  path.

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
LoRA-style decay/gate projections instead of attention.

Mitigations differ per group and are not interchangeable: percentile clipping
rescues `small` (INT6 +11.55 % → +1.60 %) but *hurts* the dense matrices,
whose outlier tail is trained signal. For dense groups the current presets
use group-wise asymmetric scales (see [Format](#format)); the earlier
SpQR-style sparse-outlier path is retained in `calibration/` for study.

## Caveats

- Presets are calibrated on one 1.5B checkpoint; recalibrate for other sizes
  (`small`/`g_lora` stay INT8 in both presets for a reason).
- Kernel dispatch tables are tuned on an M4 base chip; other Apple Silicon
  will work but may prefer different (simdgroups x rows) configs.
- ppl deltas are measured on a small held-out corpus; treat them as relative
  quality signals, not benchmarks.
- `scripts/` and `examples/` are placeholders for now — the maintained entry
  points are `rwkv_quant.api` and the benches/gates under `tests/`.
- CUDA backend is an empty stub; Metal is the only real inference path today.

## Repo layout

See [STRUCTURE.md](./STRUCTURE.md). Session-to-session engineering log with
measurement methodology (fanless-Mac A/B discipline, bit-exactness gates)
lives in [NEXT_SESSION.md](./NEXT_SESSION.md) and git history.
