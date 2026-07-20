# rwkv-quant

Quantization toolkit **and Metal inference backend** for RWKV-7 on Apple
Silicon. Portable `.rwkvq` checkpoint format, outlier-aware calibration, and
custom Metal GEMV kernels that decode the quantized format directly — no
dequantized weight copy in memory.

Reference model: `rwkv7-g1h-1.5b` (BlinkDL G1H 1.5B, bf16 2953 MB).
Reference machine: M4 MacBook Air 16 GB (base chip, fanless).

## Results

| model | size | ppl | vs bf16 | decode | prefill (T=1024) |
|---|---|---|---|---|---|
| bf16 (no quant) | 2953 MB | 11.430 | — | — | — |
| **`reduction`** (ours — groupwise int6 + AW) | 1255.9 MB (2.35x) | 11.4438 | **+0.12%** | 17.7 ms/tok | 437 tok/s |
| **`compression`** (ours — groupwise int4/5 + AW) | 970.7 MB (3.04x) | 11.7125 | +2.47% | **14.8 ms/tok** (14.15 fused, 14.02 w/ n-gram speculation) | 545 tok/s |
| MollySophia MLX int6† (community, native MLX affine) | 1272.0 MB | 11.5507 | +1.06% | 15.10 ms/tok | 591 tok/s |
| Canonical int6 RTN‡ (naive per-row, no groups/AW) | 1530.5 MB | 13.2726 | +16.12% | 19.44 ms/tok | 630 tok/s |
| Canonical int4 RTN‡ (naive per-row, no groups/AW) | 767.1 MB | 3798.62 | ~332x (broken) | 13.65 ms/tok | 543 tok/s |

`reduction` beats the community int6 build on quality (9x less degradation)
at a comparable size. `compression` beats it on every axis: smaller, faster,
and less than half the quality loss. Against the canonical (naive,
off-the-shelf-style) baselines: `reduction` is both **smaller and 130x less
degraded** than canonical int6 despite the same nominal bit depth — canonical
int6 has no real sub-byte packing in this codebase, so it lands at roughly
the same disk size as int8 would; naive bit-width choice alone does not buy
real compression without a dedicated packer. Canonical int4 goes further and
simply breaks the model (ppl 3798, not a typo). See [Method](#method) for why.

† MollySophia hasn't published a G1H build, so her row is measured on the G1
checkpoint, not G1H — size/decode/prefill are apples-to-apples (identical
architecture, same process, same machine), but part of the ppl gap may
reflect the different base checkpoint, not only the quantization scheme.

‡ Canonical rows were measured on the same reference machine and corpus, but
as separate single-model runs (`tests/eval_canonical_int4.py` /
`eval_canonical_int6.py`), not in the same interleaved four-way A/B process
used for the bf16/MollySophia/`compression`/`reduction` row — see
[What "ppl" means](#what-ppl-means-and-how-its-measured) for why that
distinction matters for the timing columns (not for ppl, which doesn't
depend on thermal state).

## Method

Four things distinguish `reduction`/`compression` from a generic INT4/INT6
quantizer (i.e. from the canonical rows above):

1. **Groupwise asymmetric scale, not per-row/per-tensor.** Weights are split
   into blocks of 32 (each with its own 6-bit scale/min pair), grouped into
   superblocks of 256 (each with an fp16 scale pair the 6-bit pairs multiply
   against) — see [Format](#format). A per-row or per-tensor scale is set by
   that row's single largest value; real RWKV-7 weights have per-channel
   outliers **40-96x** the typical magnitude (measured on the 1.5B
   reference — see [Root cause](#root-cause)), so a per-row scale forces
   everything else in that row into 1-2 quantization codes. A block of 32 is
   small enough that an outlier only damages its own block, not the whole
   row.
2. **Real sub-byte packing.** Blocks are stored as nibbles (with extra
   bit-planes for 5/6-bit codes) — an actual `IN/2`-byte footprint for 4-bit
   codes, not codes-stored-as-int8-anyway. This is *why* canonical int6 in
   the table above is the biggest file in the comparison despite being lower
   bit-depth than bf16's implicit 16: without a dedicated packer, "6-bit"
   quantization has nowhere to shrink to below int8.
3. **Activation-weighted (AW) scale search, applied per group, not
   blanket.** Instead of minimizing raw weight reconstruction error, AW
   search minimizes error weighted by activation statistics from a
   calibration pass — it measurably helps `cmix`/`emb_head` at these bit
   depths, but *hurts* `proj` at 6-bit (a result from direct measurement,
   not something predicted in advance — see `presets.py`). Each preset turns
   AW on or off per group based on what the measurement showed, not a single
   global switch.
4. **Per-group bit-width calibration instead of a uniform bit depth.**
   Sensitivity to quantization varies enormously by parameter group *and*
   doesn't transfer across model scale — `small`/`g_lora` survive INT2 on a
   61M model and go catastrophic on the 1.5B reference at the same bit
   depth (see [Why quantization sensitivity doesn't transfer across
   scale](#why-quantization-sensitivity-doesnt-transfer-across-scale)).
   `reduction`/`compression` are presets calibrated on the 1.5B reference;
   `calibrate()` runs the same per-group search on a checkpoint of your
   choosing instead of assuming a preset transfers.

A sparse-outlier scheme (SpQR-style: store a few exact values out-of-band
instead of quantizing them) was tried earlier and is retained in
`calibration/` for reference, but groupwise scale alone reached about half
the quality loss of per-row+SpQR at the same size — see
[Format](#format) for that finding.

## What "ppl" means and how it's measured

**Perplexity (ppl)** = `exp(mean token negative log-likelihood)` under
teacher forcing — informally, how "surprised" the model is by the correct
next token on average, exponentiated back into a per-token scale. Lower is
better; a bf16 model is the reference point, and every other row is reported
as `+X%` — the relative *increase* in ppl caused by quantization (so `+0.12%`
means near-lossless, `~332x` means broken).

All numbers in this README come from the same fixed corpus: an 8-sequence x
128-token slice (1024 tokens, 1016 scored after the next-token shift) from a
~91k-character, multi-language, multi-topic text mix (literature, technical
writing, encyclopedic text, several languages), tokenized once with the
standard RWKV World tokenizer (byte-level trie, greedy longest match) and
reused unchanged across every row. **This is not a published benchmark** —
not WikiText, not LAMBADA — so the absolute ppl numbers are only meaningful
*relative to each other*, on this exact slice, with this exact tokenizer;
don't compare them to numbers from other papers or other corpora.

Every row is scored through its own real quantized kernel end-to-end (never
a dequant-to-dense shortcut) — `reduction`/`compression` via
`backends/metal/quant_linear_gw.py`, MollySophia's via `mx.quantized_matmul`
(her native MLX format), canonical via the plain per-row kernel — so the ppl
numbers reflect what you'd actually get running that kernel, not a
theoretical best case. The WKV-7 recurrence itself is never quantized in any
row; only the linear projections differ by scheme.

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

---

## Open problems / contributions welcome

A few things are known gaps, not yet done — good entry points if you want to
contribute:

- **CUDA backend is an empty stub.** Metal is the only real inference path
  today; the `.rwkvq` format itself is backend-independent, so a CUDA kernel
  implementation is "just" a kernel, not a format change.
- **LoRA-style gate branch is un-fused.** The small per-layer decay/gate
  matmuls (`w/a/v/g_lora`) currently cost ~6-8 separate kernel launches per
  layer; fusing them into one or two custom kernels is estimated at another
  ~0.5-1 ms/token on decode, not yet built.
- **The fused kernel path (`FUSE=True`) isn't the default yet**, despite
  being a stable ~0.8 ms/token win with matching correctness gates — flipping
  the default (and updating the benchmarks that assume `FUSE=False`) is
  pending.
- **Sub-nibble packing.** The current nibble container has a hard floor
  around 887 MB for this model at acceptable quality — going smaller needs a
  new on-disk format (sparsity- or sub-nibble-based), not just a bit-width
  tuning pass.
- **`calibrate()` is single-group ablation.** It picks per-group bit width
  independently and doesn't fully model interaction effects when several
  groups are quantized simultaneously; a joint search would be more accurate
  but much more expensive to run.
- **Presets are calibrated on one 1.5B checkpoint.** Validating (or
  recalibrating) `reduction`/`compression` on other model sizes — especially
  much smaller or much larger ones, where sensitivity per group is known to
  shift (see [Why quantization sensitivity doesn't transfer across
  scale](#why-quantization-sensitivity-doesnt-transfer-across-scale)) — is
  open.

## Author

Alexei Goncharov
