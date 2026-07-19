# Repo structure

```
rwkv-quant/
├── pyproject.toml
├── README.md
├── STRUCTURE.md
├── LICENSE
├── rwkv_quant/
│   ├── __init__.py
│   ├── api.py                     # high-level entry points: quantize(), calibrate()
│   ├── presets.py                 # "reduction" / "compression" QuantConfig presets
│   │
│   ├── calibration/                # platform-agnostic. pure PyTorch. no backend
│   │   ├── __init__.py             # code lives here -- this is the actual "brain"
│   │   ├── outlier_scan.py         # per-channel max/mean ratio profiling
│   │   ├── fake_quant.py           # symmetric RTN + percentile-clip + SpQR sparse
│   │   ├── group_config.py         # QuantConfig, GROUPS, bit/outlier-frac/clip fields
│   │   └── ablation.py             # perplexity ablation harness (single-group,
│   │                                #   mixed-config, bit/percentile sweeps)
│   │
│   ├── formats/                    # the portable quantized checkpoint format.
│   │   ├── __init__.py             #   this is the missing "unified format" gap
│   │   ├── schema.py                #   in the RWKV ecosystem -- per-tensor bits +
│   │   ├── writer.py                #   sparse outlier indices/values, backend-agnostic
│   │   └── reader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rwkv7_ref.py            # reference forward pass, used ONLY for
│   │   │                            #   calibration/validation (slow, exact --
│   │   │                            #   not for production inference)
│   │   └── naming.py               # detects checkpoint naming scheme (custom
│   │                                #   vs. official BlinkDL "world" naming) and
│   │                                #   normalizes to one internal representation
│   │
│   └── backends/                   # platform-specific inference. consumes
│       ├── __init__.py             #   the .rwkvq format from formats/, produced
│       ├── metal/                  #   by calibration/. this is the ONLY layer
│       │   ├── __init__.py         #   that's platform-specific.
│       │   ├── quant_model.py      # QuantRWKV7: stateful decode/prefill, fused paths
│       │   ├── quant_linear_gw.py  # sb6 GEMV/GEMM kernels (кернель-3), N-batch verify
│       │   ├── quant_linear_v2.py  # per-row packed linears (loras/small)
│       │   └── quant_linear.py     # legacy
│       └── cuda/                   # empty stub for now
│           └── __init__.py
│
├── scripts/                        # CLI wrappers -- EMPTY STUBS for now,
│   ├── calibrate.py                #   use rwkv_quant.api directly
│   ├── quantize.py
│   └── benchmark.py
│
├── tests/                          # gates (test_*), benches (bench_*),
│   ├── test_gw_kernel*.py          #   profilers, spec-decode demos, eval
│   ├── test_gw_nb_parity.py        #   harnesses; venv/ lives here.
│   ├── test_fuse_parity.py         #   Naming: гейты бит-в-бит и численные,
│   ├── bench_*.py, profile_*.py    #   бенчи только A/B в одном процессе
│   ├── spec_decode_*.py            #   (см. законы в NEXT_SESSION.md)
│   └── fixtures/                   # small synthetic checkpoints, NOT the real
│                                    #   61M/1.5B models (too large for the repo)
│
└── examples/
    └── quantize_ru60m.md           # reproducible walkthrough on a small model
```

## Design principles

**`calibration/` never imports from `backends/`.** It's the part that
actually answers "which bits, which channels stay exact" — pure PyTorch,
runs identically on a laptop or a CUDA box. This is deliberate: the
finding that motivated this repo (sensitivity doesn't transfer across
model scale) means calibration has to be re-run per checkpoint, so it
needs to be fast and portable, not tied to whichever backend you're
targeting.

**`formats/` is a first-class module, not a detail inside `backends/`.**
The RWKV ecosystem doesn't currently have a shared quantized-checkpoint
format (unlike GGUF for llama.cpp) — `rwkv_quant`'s `.rwkvq` format is
meant to be backend-independent so a checkpoint quantized on a Mac can be
served from a CUDA box without re-running calibration.

**`models/rwkv7_ref.py` is explicitly not a production inference path.**
It's a from-scratch PyTorch port used only so calibration/ablation can
run without depending on any backend. It's slow on purpose (O(T) python
loop, matches the reference math exactly) — correctness over speed,
because calibration results feed directly into the quantization decisions
shipped to backends.

**`backends/` are the only platform-specific code.** `metal/` wraps
[rwkv-metal](https://github.com/)'s kernels; `cuda/` is a separate
implementation. Both are expected to consume the same `.rwkvq` files
produced by `calibration/` + `formats/writer.py` — no backend-specific
calibration logic.

**Two entry points, one underlying config type.** `api.py`'s `quantize()`
accepts either a `preset` string (`"reduction"` / `"compression"`, defined in
`presets.py`) for quick start, or an explicit `QuantConfig` for full
control. Both paths end up as the same `QuantConfig` object consumed by
`calibration/fake_quant.py` — presets are just pre-filled configs, not a
separate code path.
