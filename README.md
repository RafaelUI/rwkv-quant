# rwkv-quant

Quantization toolkit **and Metal inference backend** for RWKV-7 on Apple
Silicon. Portable `.rwkvq` checkpoint format, outlier-aware calibration, and
custom Metal GEMV kernels that decode the quantized format directly — no
dequantized weight copy in memory.

Reference model: `rwkv7-g1h-1.5b` (BlinkDL G1H 1.5B, bf16 2953 MB).
Reference machine: M4 MacBook Air 16 GB (base chip, fanless).

## Results

Quality on a **multilingual held-out corpus** (38 x 512 tokens = 19 456
predictions; Russian / English / Serbian), scored end-to-end through the
real quantized kernel. Speed on an M4 MacBook Air 16 GB, fanless.

### 1.5B (`rwkv7-g1h-1.5b`, bf16 ppl 8.1980, 3055 MB)

| build | size | Δppl vs bf16 | en / ru / sr | decode | prefill |
|---|---|---|---|---|---|
| **`reduction`** (Q6_K-style sym, 8-bit proj/head/emb) | 1435.1 MB (2.13x) | **+0.108%** | +0.18 / +0.01 / +0.25% | 17.00 ms/tok | **734 tok/s** |
| **`compression`** (groupwise int4/5) | 970.2 MB (3.15x) | **+3.63%** | +4.01 / +3.40 / +3.78% | **13.15 ms/tok** | 612 tok/s |

### 2.9B (`rwkv7-g1h-2.9b`, bf16 ppl 7.1630, 5896 MB)

| build | size | Δppl vs bf16 | en / ru / sr |
|---|---|---|---|
| **`reduction`** | 2736.8 MB (2.15x) | **+0.157%** | +0.27 / +0.04 / +0.32% |
| **`compression`** | 1854.7 MB (3.18x) | **+4.24%** | +5.69 / +3.05 / +5.46% |

### Against llama.cpp (1.5B, same data split)

Calibration data (our activation stats / their imatrix) and evaluation
data are split identically for both systems. Absolute ppl is not
comparable across the two (llama.cpp tokenizes the stream differently),
so both are reported as Δ% against their own FP baseline.

| | size | Δppl | | | size | pp512 | tg128 |
|---|---|---|---|---|---|---|---|
| `reduction` | 1435 MB | **+0.108%** | | `compression` | 970 MB | 633.2 t/s | **75.3 t/s** |
| Q6_K + imatrix | 1336 MB | +0.19% | | Q4_K_M + imatrix | 990 MB | **745.2** | 58.2 t/s |
| `compression` | 970 MB | +3.63% | | `reduction` | 1435 MB | **771.4** | **58.2 t/s** |
| Q4_K_M + imatrix | 990 MB | +3.44% | | Q6_K + imatrix | 1336 MB | 705.5 | 48.6 t/s |

At the near-lossless point we are now **ahead on all three axes at once**:
`reduction` is +0.108% against Q6_K+imatrix's +0.19%, decodes **20%
faster** (58.2 vs 48.6 t/s) and prefills **9% faster** (771.4 vs 705.5
t/s), at a 7% larger file. `compression` decodes **29% faster** than
Q4_K_M at comparable quality, and still loses prefill by 1.18x.

> **Earlier versions of this README said "prefill we lose roughly 2x".**
> Two separate errors were hiding there. `mx.compile` was never called on
> the prefill path (fixing that alone was +35%), and 60% of what was left
> turned out to be the *dequantization*, not the matmul — it was written
> as a chain of MLX ops holding full-size intermediates and ran at
> 4.4 GB/s. One Metal kernel took it from 10.68 ms to 0.84 ms per layer.
> Neither was a modelling insight; both were found by decomposing a
> number instead of trusting it.

Without imatrix we look like winners (Q6_K +0.41%, Q4_K_M +7.65%) — the
entire margin was explained by us having activation-aware scale search
and them not. Reporting only that comparison would have been dishonest.

### Against the machine's real ceilings (both measured, neither from a datasheet)

Decode is bound by **memory bandwidth**, prefill by **arithmetic**, and
the two ceilings are different numbers that have to be measured
separately. The bus is advertised at 120 GB/s and the GPU at rather more
TFLOPS than we see; both figures are arithmetic, not measurements.
Streaming reads top out at **104 GB/s** (`tests/bench_membw.py`, two
independent instruments) and `mx.matmul` at **2.80 TFLOP/s**
(`tests/bench_gemm_peak.py`, cold machine).

| | bound by | work per call | achieved | share of ceiling |
|---|---|---|---|---|
| `compression` decode | bandwidth | 878 MB/token | 66.8 GB/s | 64% |
| `reduction` prefill (pp512) | arithmetic | 1.426 TFLOP | 2.15 TFLOP/s | 77% |

(Per-token traffic is quoted only where it has been measured directly;
it is not the file size, because `emb` is a gather of one row and the
in-memory layout is not the on-disk one.)

Prefill's remaining 23% is no longer the matmul: it decomposes into the
WKV scan (7.7%), dequantization (5.5%) and the LoRA branches (5.3%).
The WKV recurrence is still the largest single item, and it is the one
place where the arithmetic ceiling does not help at all — the scan is
sequential by construction.

> **The WKV scan used to be 12.7% of prefill and is now 7.7%**, for a
> bit-identical output (`tests/test_wkv_infer_parity.py`: 9 shapes across
> three model scales, `max|Δ| = 0`). The inference kernel was reading each
> row of `a/w/k/b/r` once per thread rather than once per threadgroup —
> 1.34 GB of loads per call against 29.4 MB of useful traffic. Staging
> them in threadgroup memory is worth 39 ms of prefill on 1.5B and 50 ms
> on 2.9B and costs nothing, because it changes neither the arithmetic
> nor the summation order. The same optimization had been sitting in the
> *training* kernel next door, with a comment saying it was worth half
> that kernel's runtime; it simply never crossed over. Two parallel
> implementations of one idea do not share fixes.

> **Earlier versions of this README reported +0.12% and +2.47%.** Those
> numbers came from an 8-sequence x 128-token slice of short English
> text — 1016 predictions. That slice understates degradation **sixfold**
> against the same corpus in full, and tenfold against the multilingual
> one at 512 tokens. The measurement was real; the corpus was flattering.
> Measuring on a convenient sample is worse than not measuring, because
> it produces false confidence. See
> [What "ppl" means](#what-ppl-means-and-how-its-measured).

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
as `+X%` — the relative *increase* in ppl caused by quantization (so `+0.8%`
means near-lossless, `+300x` would mean broken).

Quality numbers come from `eval_corpus_multiling.pt`: 38 sequences x 512
tokens = 19 456 scored predictions, split 20 Russian / 9 English / 9
Serbian, tokenized once with the standard RWKV World tokenizer
(byte-level trie, greedy longest match) and reused unchanged across every
row. Activation statistics for the AW modes are collected on **held-out
chunks** — never on the text the perplexity is measured on. **This is not
a published benchmark** — not WikiText, not LAMBADA — so absolute ppl is
meaningful only *relative to other rows here*, on this exact corpus, with
this exact tokenizer.

**The corpus decides more than the quantization scheme does.** This is
the single most expensive lesson in the project, so it is stated plainly
rather than buried. The same `reduction` build measures:

| corpus | predictions | Δppl |
|---|---|---|
| English, 8 x 128 | 1 016 | +0.12% |
| English, 24 x 128 | 3 048 | +0.74% |
| multilingual, 128 tok | — | +1.52% |
| multilingual, 512 tok | 19 456 | **+2.36%** |

Same weights, same kernel, same machine — a 20x spread driven entirely by
sample size, language mix and context length. Degradation also *grows
with context*, which a short-context corpus cannot see at all: quantization
error in the channel-wise modulators inside the recurrence does not spoil
one prediction, it distorts the state update and accumulates along the
sequence. (`reduction` is +2.36% here and +0.108% in the table above:
that row is a later preset — the `small=16` fix, the `sym` block layout,
and 8-bit `proj`/`head`/`emb`. Same corpus, same kernel.)

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

# near-lossless: 2.1x smaller, +0.11% ppl on the 1.5B reference
quantize("model.pth", "model.rwkvq", preset="reduction")

# 3.2x smaller, +3.63% ppl, fastest decode
quantize("model.pth", "model.rwkvq", preset="compression")
```

Both presets use activation-weighted (AW) scale search and expect activation
statistics at the path set in `QuantConfig.act_stats_path`
(`tests/collect_act_stats.py` produces them in ~30 s).

**If that file is missing, the AW modes silently degrade to their
non-weighted variants** — you still get a valid checkpoint of the same
format, just quantized slightly differently. The default path lives in
`/tmp` and does not survive a reboot, so the same call can produce two
different files on two different days. Pass an explicit
`act_stats_path`, or set it to `None`, if you need the result to be
reproducible.

Inference (Metal / MLX). Use `model.step`, not the raw
`model.forward_stateful`: `step` is the `mx.compile`-wrapped entry point
and its cache is keyed by shape, so the same object serves both a 512-token
prefill and single-token decode. Calling the raw function costs 35% on
prefill and 17% on decode:

```python
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal.quant_model import QuantRWKV7

model = QuantRWKV7(load_raw("model.rwkvq"))
state = model.init_state(1)
logits, state = model.step(token_ids, state)   # prefill AND decode
```

Presets are calibrated on `rwkv7-g1h-1.5b` — see
[Why presets aren't universal](#why-quantization-sensitivity-doesnt-transfer-across-scale).
For a checkpoint-specific config run `calibrate()` or build a `QuantConfig`
by hand (per-group bits, group scale sizes, scale modes, clipping).

## Format

`.rwkvq` stores two block layouts, chosen per parameter group.

**`sym`** (Q6_K-style, what `reduction` uses for `proj`/`cmix`/`emb`/`head`):
blocks of **16** weights share one **int8** scale, superblocks of 16 blocks
share an fp16 `d` that those int8 scales multiply. No per-block minimum at
all. This costs 6.5625 bits/weight at 6-bit codes and 8.5625 at 8-bit —
against `sb6`'s 6.5 — and buys a factor of 75 on `cmix`: at six bits a
separate min is paid for twice (six bits for the min *and* a scale
truncated to six bits), while halving the block and giving the scale a
whole byte spends the same budget better.

**`sb6`** (group-wise asymmetric, what `compression` uses and what the LoRA
branches use in both presets): blocks of 32 weights share a 6-bit scale/min
pair (`qs`/`qm`), superblocks of 256 share an fp16 pair (`d`/`dm`) that the
6-bit pairs multiply. Codes are packed as nibbles; INT5/INT6 add one/two
bit-planes on top. Scale search is
activation-weighted where it helps (per-group setting). The format is
backend-independent; per-tensor bits and modes live in the file, not in code.

**The container is safetensors, not pickle.** A `.rwkvq` file is a
safetensors archive: flat `key::field` buffers plus one JSON manifest in
`__metadata__`. It has no executable payload, memory-maps instead of
loading (2.9B: 0.03 s and 0.25 GB resident, against 0.24 s and 2.08 GB for
the old `torch.save` container), and reads without torch or even without
this package installed — `formats/codec.py` is a complete reader in pure
numpy, meant to be ported to Swift/C++ as-is:

```python
from rwkv_quant.formats import codec
manifest, arrays = codec.open_rwkvq("model.rwkvq")
w = codec.dequant_key(manifest, arrays, "emb.weight")   # float32
```

That import pulls in numpy and nothing else — the package's other entry
points are lazy, so `import rwkv_quant` does not drag torch in. This is
load-bearing rather than tidy: consumers of the format run in
environments that have no torch at all, and the guarantee is enforced by
a gate that blocks torch at the import machinery and then exercises the
codec (`tests/test_torch_free_import.py`).

`codec` also builds the two *loader* layouts a backend may want, from the
same file and still without torch:

```python
qblk, qsqm, ddm, xbits = codec.sb6_to_k3(...)          # Metal GEMV kernel
wq, scales, biases, bits = codec.sb6_to_mlx_affine(...) # MLX quantized_matmul
```

Both are relayouts, not requantizations — the codes, scales and biases
are the calibrated ones, byte for byte. (Calling `mx.quantize()` on a
dequantized weight would look equivalent and quietly is not: it
recomputes scale/bias from block min/max and throws the calibration
away.) This is why a `.rwkvq` no longer needs a companion sidecar file:
whatever layout the backend wants, it can build at load time.

`load_raw()` still reads checkpoints written by older versions — the two
containers are told apart by their first bytes.

The manifest is self-describing, which matters more than it sounds. Two
things used to live only in comments and would silently corrupt a
port: official ("world") checkpoints store the LoRA matrices
`w1/w2/a1/a2/v1/v2/g1/g2` transposed, and the number of quantization
blocks is `ceil(IN/gs)`, not `IN // gs` — `blocks.N.att.w1` is `[2048,
96]` at `gs=64`, so a reader that divides gets one block where there are
two and applies the wrong scale to the tail. Both are now recorded per
tensor (`transposed`, `n_blocks`), along with the full quantization
config as structured JSON.

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

## What a bit actually costs on disk

The search in `calibrate()` minimizes file size, so its cost model is
measured, not derived (`tests/probe_schema_cost.py --check` re-verifies it
against the writer):

| scheme | bits/weight | | scheme | bits/weight |
|---|---|---|---|---|
| sb6 @4 | 4.500 | | asym gw64 @5 | **9.000** |
| sb6 @5 | 5.500 | | asym gw64 @6 | **9.000** |
| sb6 @6 | 6.500 | | asym gw64 @8 | **9.000** |
| per-row RTN @4 | 4.021 | | per-row RTN @6 | **8.021** |
| | | | per-row RTN @8 | **8.021** |

Two consequences that are easy to miss:

- **The `asym` container does not shrink with bit depth.** Codes sit in
  `uint8` and scale/min are `fp32` per block of 64, so 5, 6 and 8 bits
  produce byte-identical files. Bit depth there is a *quality* knob that
  costs nothing — which also means the presets' current `w_lora=6` is
  paying for nothing.
- **Per-row RTN has no sub-byte packing above 4 bits.** `@6` and `@8` are
  the same size. This is the same trap the "canonical int6" baseline fell
  into: choosing a lower bit width buys no compression without a packer
  that can actually store it.

## Caveats

- Presets are calibrated on `rwkv7-g1h-1.5b` and re-validated on 2.9B;
  recalibrate for other sizes (`small` stays bf16 and `g_lora` stays INT8
  in both presets for a reason).
- **Sensitivity genuinely does not transfer between scales, including in
  ways that reverse a decision.** Recent example: dropping `emb` from 6 to
  5 bits is free on 1.5B (−0.07 pp, −16.8 MB) and costs +0.61 pp on 2.9B.
  Any preset change is validated on both checkpoints before it lands.
- Kernel dispatch tables are tuned on an M4 base chip; other Apple Silicon
  will work but may prefer different (simdgroups x rows) configs.
- ppl deltas are measured on one held-out corpus; treat them as relative
  quality signals, not benchmarks.
- `scripts/` and `examples/` are placeholders for now — the maintained entry
  points are `rwkv_quant.api` and the benches/gates under `tests/`.
- CUDA backend is an empty stub; Metal is the only real inference path today.
- 2.9B calibration runs need ~15.5 GB peak (measured with
  `/usr/bin/time -l`, `peak memory footprint`; RSS under-reports this by
  5x on unified memory). One config per process.

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
- **`calibrate()` screens groups in isolation.** It picks each group's
  scheme independently and then refines the composite until it fits the
  budget, but it does not model interaction effects during the search; a
  joint search would be more accurate and much more expensive.
- **Schemes are chosen per *group*, not per *tensor*.** A group is
  quantized with one scheme, so a group whose tensors have mixed shapes
  falls back to whatever works for all of them. The LoRA branches are
  exactly that case: `w2/a2/v2/g2` have `IN = n_embd` and qualify for the
  cheaper, more accurate groupwise `sb6` layout, while `w1/a1/v1` have
  `IN = rank` (96, 64) and do not — so the whole group takes the worse
  scheme. The on-disk format already records the layout **per tensor**
  (`kind` in the manifest), so this is a writer-dispatch limitation, not a
  format one.
- **Non-uniform code books (NF4/AF4-style).** Every scheme here maps codes
  to a uniform grid. A normally-distributed code book costs the same
  number of bits and only changes a 16-entry lookup table in the kernel —
  the classic "free quality" lever, and the one most likely to unlock a
  lower bit width. Not tried.
- **Presets are calibrated on 1.5B and validated on 2.9B.** Smaller and
  much larger checkpoints are open, and sensitivity is known to shift
  (see [Why quantization sensitivity doesn't transfer across
  scale](#why-quantization-sensitivity-doesnt-transfer-across-scale)).

## Author

Alexei Goncharov
