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

### Quality across four scales

Same corpus, same tokenizer, same harness for every row. `reduction` is the
near-lossless preset, `compression` trades quality for size. Sizes are the
actual `.rwkvq` files on disk.

| model | bf16 ppl | build | size | Δppl (all) | en / ru / sr |
|---|---|---|---|---|---|
| **0.1B** (`rwkv7-g1d-0.1b`) | 15.183 | `reduction` | 191.3 MB (2.00x) | **+0.29%** | +0.22 / +0.18 / +0.59% |
| | | `compression` | 133.9 MB (2.85x) | **+7.50%** | +5.22 / +7.97 / +8.77% |
| **0.4B** (`rwkv7-g1d-0.4b`) | 10.994 | `reduction` | 434.1 MB (2.08x) | **+0.18%** | +0.15 / +0.24 / +0.06% |
| | | `compression` | 301.2 MB (2.99x) | **+4.20%** | +3.43 / +4.32 / +4.72% |
| **1.5B** (`rwkv7-g1h-1.5b`) | 8.198 | `reduction` | 1439.0 MB (2.12x) | **+0.04%** | -0.05 / +0.03 / +0.17% |
| | | `compression` | 993.4 MB (3.08x) | **+2.82%** | +3.12 / +2.28 / +3.74% |
| **2.9B** (`rwkv7-g1h-2.9b`) | 7.163 | `reduction` | 2743.3 MB (2.15x) | **+0.06%** | +0.07 / +0.03 / +0.12% |
| | | `compression` | 1885.5 MB (3.13x) | **+1.85%** | +2.33 / +1.66 / +1.81% |

> **All four `compression` rows re-measured 2026-09-21: the embedding is now
> 6-bit instead of 5** (`head` stays at 5). On 2.9B the embedding carried
> 17% of the remaining KL; one more bit takes 2.9B from +2.43% to +1.85%
> (KL 0.0320 to 0.0249) for +21 MB, and 0.1B from +8.00% to +7.50% (both
> significant by paired bootstrap over the 38 windows). At 0.4B and 1.5B the
> change is inside noise. Keeping the embedding fully in bf16 would buy only
> 5% more on 2.9B at ten times the bytes. It applies unconditionally: the
> price is file size only (+1.1% at 2.9B, but +4.9% at 0.1B, where the
> vocabulary is a large share of the model); in memory the `compression`
> embedding is expanded to fp16 anyway and read one row per token, so RAM
> and decode speed do not change. Each new file was measured next to the
> previous one with the same instrument; the previous rows reproduce to
> +-0.01 points.
>
> **All eight rows re-measured 2026-09-20: layer-0 `o_proj` now stays in
> bf16 in both presets.** Leave-one-out on the real path showed that the 2.9B
> `compression` deficit (+4.04% against +2.85% at 1.5B) was mostly one matrix:
> the output projection of layer 0, whose input carries a massive activation
> channel (one channel ~7e6 times the median energy). Keeping that single
> matrix unquantized takes 2.9B `compression` from +4.04% to +2.43% (KL 0.052
> to 0.032) for +9.4 MB, and 0.1B from +8.68% to +8.01%. At 0.4B and 1.5B the
> change is inside noise, and `reduction` barely moves (8-bit weights already
> cope). It costs 0.2-0.7% of file size and applies unconditionally, as
> insurance: the activation statistic does not predict which models benefit
> (0.1B gains with a mild channel, 0.4B and 7.2B gain nothing with strong
> ones). Decode is not slower (2.9B, interleaved A/B: -0.4 to -0.5 ms/token;
> A/A control ±0.02). All rows are now measured from the written `.rwkvq`
> files with one instrument (38 windows, fp32 reference as the ppl base); the
> older `reduction` rows came from an earlier recipe, which is why they move
> more than the change itself explains. The KL tables further down predate
> this change.
>
> **All four `compression` rows re-measured 2026-09-18.** The bit allocation
> was changed on 09.09 (`proj` 5 to 4, `cmix.key` 4 to 5) after per-matrix
> sensitivity turned out to diverge several-fold inside the old groups, and
> only the 1.5B row had been recomputed. All four are now measured with one
> recipe: calibration statistics from the repo corpus (the shipped default,
> `act_stats=auto`), evaluation on the same 38 windows, real sb6 packing.
> The 1.5B number reproduced the 09.09 measurement exactly (+2.850%). Sizes
> are unchanged: the reallocation moved bits, it did not add them.
> `reduction` rows were not re-measured and are unaffected by it.
>
> Calibration corpus matters more than 09.09 assumed, and not in one
> direction. Against a second honest corpus (three languages, disjoint from
> the evaluation half, 8704 tokens against 11776 in the repo one) the deltas
> move to +8.30 / +3.87 / +3.22 / +4.69%: the narrow domain-matched corpus
> wins on the two small models by about 0.4 points and loses on the two big
> ones by as much. The flip lines up with size and with the checkpoint
> series (g1d below, g1h above) at once, and four points cannot separate
> those. Both corpora are disjoint from what is scored: this is sensitivity
> of the activation-weighted search, not leakage.
>
> **Volume or domain, and does it matter? (2026-09-19, 0.4B and 1.5B.)** The
> narrow corpus is also smaller (8687 tokens against 11753), so a third arm
> takes the repo corpus down to the same volume without changing its mix
> (every window shortened, none dropped). On the real path, from the same
> files: volume alone moves ppl by +0.12 / -0.09 points (0.4B / 1.5B),
> domain at equal volume by +0.29 / -0.27, both inside their paired 95%
> intervals; only the full repo-versus-narrow gap at 0.4B (0.42 points) is
> significant. So most of the shift is domain, it keeps its scale-dependent
> sign at equal volume, and it is small. **KL does not see it at all**: all
> three calibrations sit within 1.5% of each other and every pair is inside
> its interval. The choice of calibration corpus moves which tokens the
> quantized model happens to favour, not how far it drifts from the fp
> model, so the shipped default stays.

Perplexity is a coarse instrument at these margins, so `reduction` is also
scored by **KL divergence against an fp32 reference** — same weights, same
inputs, activations in fp32 — on 8 x 512 tokens through the real quantized
kernel. Those 8 windows are the first 8 of the evaluation corpus and are
**all Russian**, so this table is a Russian-only reading, unlike the ppl
columns above. This separates rows that ppl cannot:

| model | KL vs fp32 (nats/token) | 95% CI | top-1 agreement |
|---|---|---|---|
| 0.1B | 0.006301 | [0.005889; 0.006746] | 94.84% |
| 0.4B | 0.003637 | [0.003397; 0.003937] | 95.99% |
| 1.5B | 0.002274 | [0.002043; 0.002538] | 97.31% |
| 2.9B | 0.001661 | [0.001496; 0.001854] | 97.80% |

KL falls monotonically with scale: bigger models absorb the same quantization
scheme better. Δppl does **not** follow that order (0.33 / 0.33 / 0.15 /
0.24%), which is a fair warning about reading too much into a tenth of a
percent of perplexity on 19 456 predictions. The two metrics agree on the
big picture and disagree on the fine ordering; where they disagree, KL is
the more sensitive of the two.

`compression` scored the same way (measured 2026-09-19), but on **all 38
windows** — the same 19 418 predictions and the same en / ru / sr split as
the ppl columns — through the same `.rwkvq` files listed above:

| model | KL vs fp32 (nats/token) | 95% CI | top-1 agreement | en / ru / sr |
|---|---|---|---|---|
| 0.1B | 0.089564 | [0.081614; 0.097300] | 81.67% | 0.0547 / 0.0952 / 0.1119 |
| 0.4B | 0.049644 | [0.045884; 0.053345] | 86.52% | 0.0356 / 0.0506 / 0.0614 |
| 1.5B | 0.035989 | [0.033966; 0.038189] | 89.14% | 0.0362 / 0.0352 / 0.0376 |
| 2.9B | 0.052052 | [0.047391; 0.057174] | 89.00% | 0.0496 / 0.0506 / 0.0578 |

Here KL and ppl agree on the ordering, including the one that breaks the
trend: `compression` is best at 1.5B and worse again at 2.9B, where
`reduction` keeps improving with scale. The preset was tuned on 1.5B
(see below). On the Russian-only 8-window slice used for `reduction` the
`compression` rows read 0.1067 / 0.0556 / 0.0365 / 0.0487, so the two
tables are directly comparable only through that slice: `compression`
costs 15-17x the KL of `reduction` at 0.1B-1.5B and 29x at 2.9B.

`compression` degrades far faster on small models than `reduction` does:
+8.68% at 0.1B against +0.33%. Presets in this repo were tuned on the 1.5B
checkpoint, and the numbers above are the measured cost of assuming they
transfer.

#### These numbers replace earlier ones

Two things changed on 2026-08-28, and both moved every row:

1. **LoRA is now quantized along the reduction axis.** The writer used to
   quantize the LoRA matrices in the checkpoint's raw `[in, out]` layout,
   which put the quantization groups along the *output* axis of the matmul.
   Grouping along the reduction axis instead is worth **KL 0.003303 ->
   0.002101** and **Δppl +0.24% -> +0.15%** at 1.5B, for 192 extra bytes.
2. **Activation statistics are now collected on a corpus disjoint from the
   evaluation corpus.** They previously were not, despite what this file
   claimed. Calibrating on the text the perplexity is measured on flattered
   `reduction` by about 0.13 points and `compression` by about 0.5 points at
   1.5B.

Measured at 1.5B, all four corners, so neither effect has to be taken on
trust:

| LoRA layout | calibration | `reduction` Δppl | `compression` Δppl |
|---|---|---|---|
| output axis (old) | eval corpus (leaked) | +0.11% | +3.63% |
| reduction axis (new) | eval corpus (leaked) | +0.02% | +3.52% |
| output axis (old) | held-out (honest) | +0.24% | +4.16% |
| reduction axis (new) | held-out (honest) | **+0.15%** | **+4.03%** |

The gain from the layout change is the same (-0.09 points) under either
calibration, so it is real and not an artifact of the statistics. The
previously published headline of `+0.108%` for `reduction` at 1.5B was the
top-left cell of that table: correct arithmetic on a leaky experiment.

### Against llama.cpp (1.5B)

RWKV-7 support in llama.cpp is MollySophia work, so these tables compare
against it. llama.cpp comes from Homebrew (Metal + BLAS backends, 4 threads)
and runs on the same machine. Its GGUF files are converted from the same
`.pth`, with an imatrix taken from calibration text disjoint from the
evaluation corpus. Absolute ppl is not comparable across the two (the
streams are tokenized differently), so quality is reported as a delta
against each side FP baseline. Backends not covered: `rwkv-mobile` (same
author, Apple-targeted) and `web-rwkv` (wgpu); neither is installed here.

#### Quality (2026-09-09, disjoint calibration on both sides)

Evaluation on 38 x 512 tokens; the llama.cpp deltas are paired
(`--kl-divergence-base`) and the interval is on the ppl ratio. Sizes are
decimal MB on both sides.

| | size | Δppl |
|---|---|---|
| `reduction` | 1435.1 MB | +0.153% |
| Q6_K + imatrix | 1336.1 MB | +0.235% ± 0.096 |
| `compression` | 970.2 MB | +2.850% |
| Q4_K_M + imatrix | 990.1 MB | +3.147% ± 0.317 |

At the near-lossless point the two are indistinguishable (our delta is
inside one of their sigmas) and our file is 7.4% larger. At the compression
point `compression` is smaller and 0.3 points better, also inside their
interval. Across the size range between the two, our Pareto front lies
above theirs: at 1062.5 MB we measure +1.657% where interpolating Q4_K_M
and Q6_K gives about +2.54%.

#### Prefill, pp2048 (2026-09-17)

`llama-bench -p 2048 -n 0 -ngl 99 -r 5 -b 2048` (build 10150) with ubatch
512, its default, and 2048, the whole prompt in one graph as on our side.
Ours: one compiled `forward_stateful` call at T=2048, logits for the last
position only, median of 5 rounds, dequant kernel on. Runs were strictly
alternated and the first pair was repeated at the end; on battery, low-power
mode off, no swap.

| pair (by size) | ours, t/s | llama.cpp ubatch 512 / 2048, t/s | ours ahead |
|---|---|---|---|
| `compression` / Q4_K_M | 783.6 | 755.8 / 754.6 | +3.7% |
| `reduction` / Q6_K | 775.7 | 731.4 / 709.2 | +6.1% |
| `compression` / Q4_K_M, repeat | 787.0 | 714.3 / 719.2 | +9.4% |

The machine warmed up during the run: llama.cpp lost 5% on the repeat while
our number did not move, so the honest reading is ahead by 4-9%,
conservatively 4%. Our throughput does not drop from T=512 to T=2048
(780 to 784 t/s), and a larger ubatch does not help llama.cpp.

> **Earlier versions of this README said "prefill we lose roughly 2x".**
> Two separate errors were hiding there. `mx.compile` was never called on
> the prefill path (fixing that alone was +35%), and 60% of what was left
> turned out to be the *dequantization*, not the matmul — it was written
> as a chain of MLX ops holding full-size intermediates and ran at
> 4.4 GB/s. One Metal kernel took it from 10.68 ms to 0.84 ms per layer.
> Neither was a modelling insight; both were found by decomposing a
> number instead of trusting it.

#### Across scales, pp1024 and tg1024 (2026-09-18)

Same command on their side (`-p 1024 -n 1024 -ngl 99 -r 3 -b 1024 -ub 1024`),
one compiled prefill call and 1024 single-token steps on ours, arms strictly
alternated on a cold machine on mains power. Pairs are matched by file size;
GGUF files are the published conversions of the same checkpoints (g1d for
0.1B and 0.4B, g1h for 1.5B and 2.9B).

| scale | build | size | pp1024, t/s | tg1024, t/s |
|---|---|---|---|---|
| **0.1B** | `compression` | 126.8 MB | 6728 | 364.8 |
| | `reduction` | 190.7 MB | 6539 | 302.3 |
| | Q8_0 | 211.0 MB | 5223 ± 253 | 72.5 ± 17.8 |
| **0.4B** | `compression` | 291.3 MB | 2457 | 183.2 |
| | `reduction` | 433.1 MB | 2411 | 148.9 |
| | Q8_0 | 500.1 MB | 2088 ± 2 | 49.8 ± 2.0 |
| **1.5B** | `compression` | 970.6 MB | 798 | 74.6 |
| | Q4_K_M | 990.1 MB | 748 ± 2 | 51.5 ± 0.8 |
| | `reduction` | 1435.1 MB | 792 | 58.1 |
| | Q6_K | 1336.1 MB | 740 ± 0.1 | 48.9 ± 2.3 |
| **2.9B** | `compression` | 1855.2 MB | 403 | 40.0 |
| | Q4_K_M | 1919.0 MB | 377 ± 4 | 34.2 ± 0.3 |

Prefill is ahead by 7% on the two big models and by 15-29% on the two small
ones. Decode is ahead everywhere and the margin grows as the model shrinks:
+17% at 2.9B, +45% at 1.5B, 3x at 0.4B, 5x at 0.1B. That shape says their
decode carries a fixed per-token cost of roughly 13-19 ms that arithmetic
hides only on large models; their own spread on the small models (± 17.8
t/s at 0.1B) points the same way.

A second pass ten minutes later, on a machine warmed by the first, moved
both sides down together (ours 403 to 325 t/s at 2.9B, theirs 377 to 294;
their Q6_K prefill 740 to 624). The table above is the cold pass. On a
fanless laptop any single number here is worth less than the alternation
that produced it.

#### Decode and pp512 (2026-09-16)

Same machine and GGUF artifacts as above,
llama.cpp from Homebrew (ggml 0.17.0, Metal + BLAS backends, 4 threads),
`llama-bench -p 512 -n 128 -r 3`. Our side: prefill is a single
`forward_stateful` call at T=512 with logits for the last position only,
which is what llama.cpp also computes during prompt processing; median of
7 rounds with the first call discarded. Quality columns are deliberately
not restated here — nothing in this row changed the numerics, the dequant
kernel is bit-exact.

| | size | pp512 | tg128 |
|---|---|---|---|
| `compression` + dequant kernel | 970 MB | **754.5-778.7** | **76.6** |
| Q4_K_M + imatrix | 943 MiB | 750.8 ± 1.9 | 60.9 ± 3.0 |
| `reduction` | 1435 MB | **764.1** | 58.2 |
| Q6_K | 1.24 GiB | 729.0 ± 12.3 | 50.4 ± 1.8 |

At pp512 `compression` used to trail Q4_K_M by 1.18x and now brackets it.
The range 754.5-778.7 is not a spread of one measurement: it is the same
benchmark on a cold machine and on a machine warmed by the llama.cpp run
before it, and llama.cpp got the cold machine, so this run alone supports
parity on prefill, not a win. The strictly alternated pp2048 run above is
the better-controlled prefill comparison. Decode is a win by a margin well
outside the noise: +26 % against Q4_K_M at a smaller file, and `reduction`
is +15 % against Q6_K.

One byproduct worth recording, from a single pair of runs rather than a
sweep: under thermal load the dequant article grew 47.4 → 66.0 ms (+39 %)
while the floor with dequant removed barely moved (610.1 → 612.6 ms). On
this fanless machine the memory-bound part of prefill degrades first and
the GEMM does not, so benchmarks that mix the two are worth running cold.

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

What is left of prefill above the GEMM floor was decomposed for `compression`
in one process on 2026-09-17 (`tests/_sess/bench_prefill_rest_ab.py`, T=512,
657 ms): dequantization 47.9 ms (7.3%), WKV scan 31.2 ms (4.7%), LoRA
branches 28.7 ms (4.4%), norms and the fused WKV tail about 9 ms, lerps,
channel-mix activation, embedding and casts about 30 ms, and about 29 ms of
interaction between items. No single item is worth more than about 2.4% of
prefill if halved. Two structural levers were measured and rejected:
batching the LoRA branches is 12 ms slower and costs 57 MB, and a chunked
DPLR form of the scan, prototyped for training, lost on this GPU (the scan
already runs 2048 threads per step, so parallelism over the sequence adds
nothing). Every ablation stub in that tool reads all inputs of what it
replaces: MLX does not compute outputs nothing depends on, and a stub that
ignores its inputs silently removes the projections feeding it too. An older
decomposition of `reduction` (August, a different instrument) gave WKV 7.7%,
dequantization 5.5% and LoRA 5.3%.

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
row. Activation statistics for the AW modes are collected on a **held-out
corpus** — never on the text the perplexity is measured on. Numbers
published before 2026-08-28 did not honour that rule; see the note under
Results for the measured size of the difference. **This is not
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
sequence. (`reduction` is +2.36% here and +0.108% on the multilingual corpus, both
under the pre-2026-08-28 setup:
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

# near-lossless: 2.1x smaller, +0.15% ppl on the 1.5B reference
quantize("model.pth", "model.rwkvq", preset="reduction", tokenizer=tok)

# 3.2x smaller, +4.03% ppl, fastest decode
quantize("model.pth", "model.rwkvq", preset="compression", tokenizer=tok)
```

`tokenizer` is required, and not for metadata. Both presets use
activation-weighted (AW) scale search: the scale for each group is chosen
by an error weighted with the mean square activation of the input
channels. Collecting that statistic means running the model over some
text, and the text has to be split with the same vocabulary the
checkpoint was trained on — which is a property of your model, not of
this library. Pass anything with an `.encode` method, a callable, or a
path to a vocabulary file.

Everything else is automatic: `quantize()` tokenizes the calibration
corpus shipped in `rwkv_quant/data/calib_corpus.txt`, collects the
statistics itself and caches them under `~/.cache/rwkv-quant`. Budget
about six minutes for a 1.5B checkpoint on the first call and nothing on
later ones. The collection pass needs the dense model in memory (~3 GB
for 1.5B, ~6 GB for 2.9B); it runs before quantization and frees the
model afterwards, so the process peak is the larger of the two steps, not
their sum.

Why this is not optional. Without the statistics, AW degrades to an
unweighted search, and measured on the 1.5B reference that costs 38% in
KL divergence from the bf16 model (0.004021 vs 0.002908 nats/token) and
0.78 points of top-1 agreement. You can still ask for it explicitly with
`act_stats=None`, and you can supply your own file with
`act_stats="/path/to/stats.pt"`.

The calibration corpus is small on purpose but wide on purpose too. The
statistic saturates at about a thousand tokens — 2, 4 and 17 sequences
give the same result within 2% — so size is not the lever. Coverage of
writing systems is: calibrating on Russian, English and Serbian and then
running Chinese or source code measures 0.004096 KL, which is no better
than having no statistics at all (0.003783). Adding Chinese and code to
the calibration brings that to 0.002888 and costs the original languages
nothing. If your models will see a script the shipped corpus does not
cover, extend it.

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

- **Single-pass prefill dequant kernel** (`backends/metal/gw_dequant_kernel.py`).
  The GEMV path above never materializes weights, but the prefill dense path
  has to hand `mx.matmul` an fp16 tensor, and expressing that materialization
  in MLX costs 175.6 ms of a 785.7 ms prefill (1.5B, T=512, single call).
  Two measured reasons: the `concatenate` that doubles the code bytes breaks
  operator fusion and costs 68.9 ms by itself, and the remaining fused chain
  runs at 48.8 GB/s against ~107 GB/s measured on the same machine. One
  thread per (row, block) — read 16-24 B, unpack in registers, write 32
  halves — takes it to 47.4 ms, i.e. prefill 785.7 → 657.5 ms and
  651.7 → 778.7 tok/s (-16.3 % time, +19.5 % throughput). The floor with the
  dequant removed entirely is 610.1 ms, and `mx.matmul` alone accounts for
  539.3 ms of it at 93 % of this machine fp16 GEMM ceiling, so what is left
  to win on prefill is small and lies outside the GEMM.

The kernel is the default prefill path since 2026-09-17 for every K3 tensor
with `xbits <= 1` (all tensors of all four scales); `xbits = 2` falls back to
the MLX chain until a real 6-bit file exercises it, and `RWKVQ_GW_DQ_REF=1`
restores the chain for A/B. The gate below also checks that the production
path actually routes every eligible tensor to the kernel.

Bit-exactness of that kernel is checked by
`tests/test_gw_dequant_kernel_parity.py`, which compares the uint16 bit
patterns of every dequantized tensor against the reference path, on real
files rather than synthetic shapes, and which is itself verified by mutation
(`MUTATE=1` corrupts one bit-plane shift and the gate must go red):

| scale | tensors | xbits 0 / 1 | mismatching elements |
|---|---|---|---|
| 0.1B | 73 | 60 / 13 | 0 |
| 0.4B | 145 | 120 / 25 | 0 |
| 1.5B | 145 | 120 / 25 | 0 |
| 2.9B | 193 | 160 / 33 | 0 |

Because the dequantized tensors are identical bit for bit, every quality
number in this README carries over to the kernel path unchanged; the
end-to-end prefill benchmark confirms it independently, with identical logit
fingerprints across all arms. Two caveats worth stating: no preset in use
produces `xbits=2` tensors, so that branch of the kernel is written but not
exercised by any of the four gates above; and reaching bit-exactness
required blocking the fma contraction — Metal folds `q * s + m` into a
single-rounding fma while MLX rounds twice, which showed up as a 1 ULP
difference on 42 % of elements until the product was rounded explicitly.

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
