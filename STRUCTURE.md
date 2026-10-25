# Repo structure

```
rwkv-quant/
├── pyproject.toml                  # rwkv-quant 0.3.0; mlx and rwkv-metal on macOS only
├── README.md                       # results, method, quick start, format, kernels
├── STRUCTURE.md
├── NEXT_SESSION.md                 # working notes (state, measurements, open items)
├── LICENSE
├── k3_patch.py                     # history: one-off patch that added kernel 3 to
│                                   #   quant_linear_gw.py; already applied, refuses to rerun
├── rwkv_quant/
│   ├── __init__.py                 # lazy imports: `formats.codec` must import without torch
│   ├── api.py                      # quantize(checkpoint, output, preset | config), calibrate()
│   ├── presets.py                  # "reduction" / "compression" as ready QuantConfigs
│   ├── world_tokenizer.py          # RWKV World tokenizer (trie), pure Python
│   │
│   ├── calibration/                # platform-agnostic, pure PyTorch -- the "brain"
│   │   ├── group_config.py         # QuantConfig, GROUPS (quantized parameter groups)
│   │   ├── fake_quant.py           # symmetric RTN, percentile clip, SpQR sparse outliers
│   │   ├── groupwise.py            # block-wise schemes (sb6, sym) in fake form
│   │   ├── gptq.py                 # GPTQ on the sb6 and sym grids
│   │   ├── act_stats.py            # E[x^2] per input channel, collected in the library
│   │   ├── autopick.py             # "weak spot -> bit" rule, both directions
│   │   ├── schema_space.py         # which schemes apply to which tensor
│   │   ├── outlier_scan.py         # per-channel max/mean profiling
│   │   └── ablation.py             # perplexity ablation harness (groups, bits, clips)
│   │
│   ├── formats/                    # the portable .rwkvq format
│   │   ├── codec.py                # NORMATIVE layout in numpy, no torch: container,
│   │   │                           #   bit layout, dequant of every layout; ports
│   │   │                           #   (rwkv-metal, SwiftRWKV) follow this file
│   │   ├── schema.py, reader.py    # torch twins of codec (faster dequant; gate
│   │   │                           #   tests/test_codec_parity.py)
│   │   ├── writer.py               # real (not fake) quantization + save to .rwkvq
│   │   └── export_mlx.py           # torch-free MLX sidecar: safetensors + JSON manifest
│   │
│   ├── models/
│   │   ├── rwkv7_ref.py            # reference forward (PyTorch), for calibration and
│   │   │                           #   validation only -- not an inference path
│   │   └── naming.py               # custom vs official BlinkDL tensor naming
│   │
│   ├── data/                       # bundled calibration text (calib_corpus.txt,
│   │                               #   gptq_calib.jsonl)
│   │
│   └── backends/                   # the only platform-specific code
│       ├── metal/                  # MLX inference over .rwkvq (macOS)
│       │   ├── quant_model.py      # QuantRWKV7: model from .rwkvq, prefill and stateful
│       │   │                       #   decode; WKV kernels come from rwkv-metal
│       │   ├── generate.py         # generation with an async_eval pipeline
│       │   ├── quant_linear_gw.py  # sb6 ("compression"): GEMV/GEMM kernels, kernel 3
│       │   ├── quant_linear_sym.py # sym ("reduction", Q6_K-like): GEMV kernels
│       │   ├── gw_dequant_kernel.py# one-pass sb6 dequant kernel -> fp16
│       │   ├── fused_prewkv.py     # pre-WKV block (w, a, kk, k/v update) in one function
│       │   ├── fused_tail.py       # TMix tail (group norm + bonus + gate) in one kernel
│       │   ├── quant_linear_v2.py  # int8 per-row scale GEMV (older format)
│       │   ├── quant_linear.py     # v1 int8 + SpQR linear, kept as the reference for v2
│       │   └── gw_linear_tiled.py  # stub: tiled GEMM not written yet; for now an alias
│       │                           #   of the dense path (the reference it must match)
│       └── cuda/                   # empty: no CUDA inference backend yet (quantize()
│                                   #   and calibration do run on CUDA, multi-GPU)
│
├── docs/
│   ├── LAWS.md                     # measurement methodology rules
│   ├── ARTICLE-1.md                # article draft: what breaks across model sizes
│   ├── VS-GGUF-MLX.md              # .rwkvq vs GGUF and MLX (numbers dated 08.10)
│   ├── TOOLS.md, REFERENCE.md      # history: QLoRA over .rwkvq, old preset tables
│   └── LOG-*.md                    # session logs split out of NEXT_SESSION.md
│
├── tests/                          # flat: test_* -- gates (bit-exact and numeric,
│   │                               #   64 files); bench_* -- A/B benches in one process;
│   │                               #   eval_*, probe_*, diagnose_*, profile_*, spec_*,
│   │                               #   ablate_* -- instruments
│   ├── mutate_streaming_gate.py    # mutation cap for test_streaming_inference
│   ├── _sess/                      # one-off session instruments (named in NEXT_SESSION)
│   ├── dev/                        # MLX dequant prototypes, reference dumps
│   ├── ane_probe/                  # Swift probes of Core ML / ANE
│   ├── data/                       # eval sets and calibration measurements (JSON)
│   └── fixtures/                   # small golden files; real checkpoints are not
│                                   #   in the repo
│
├── tests_local/                    # local helpers: rebuilding checkpoint names,
│                                   #   quantizing 0.4B
├── tools/
│   └── convert_pth_to_gguf.py      # RWKV-7 .pth -> GGUF (F16) for llama.cpp comparisons
│
└── scripts/                        # calibrate.py, quantize.py, benchmark.py -- empty
                                    #   stubs; use rwkv_quant.api
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

**`formats/codec.py` не импортирует torch — и это не стилистика.**
Потребители формата (`rwkv-metal` на MLX, `SwiftRWKV`) torch-free, и до
сих пор портировали раскладку с докстрингов, сверяясь с эталоном руками.
`codec.py` — нормативная реализация формата ЦЕЛИКОМ на numpy: разбор
контейнера safetensors через mmap, битовая раскладка, деквант всех трёх
квантованных раскладок. Порт делается с одного файла:

```python
manifest, arrays = codec.open_rwkvq("model.rwkvq")
w = codec.dequant_key(manifest, arrays, "emb.weight")   # float32
```

`schema.py` и `reader.py` держат torch-двойники **осознанно**: замер
(`tests/bench_codec_dequant_ab.py`) показал 2.6x на деквант в пользу
torch, потому что он параллелит поэлементные операции по ядрам.
Расхождение двух реализаций ловит `tests/test_codec_parity.py` — при
правке раскладки править оба файла и гонять гейт.

**Контейнер `.rwkvq` — safetensors, не pickle.** Плоские буферы
`ключ::поле` плюс манифест одним JSON в `__metadata__`. Прежний
`torch.save` требовал torch И установленного `rwkv_quant` той же версии
(pickle пишет полные имена классов), исполнял код при загрузке и читал
файл целиком в анонимную память. Новый контейнер отображается: 2.9B
грузится за 0.03 с при 0.25 ГБ maxrss против 0.24 с и 2.08 ГБ.
`load_raw` читает оба — контейнеры различаются по первым байтам.

**`models/rwkv7_ref.py` is explicitly not a production inference path.**
It's a from-scratch PyTorch port used only so calibration/ablation can
run without depending on any backend. It's slow on purpose (O(T) python
loop, matches the reference math exactly) — correctness over speed,
because calibration results feed directly into the quantization decisions
shipped to backends.

**`backends/` are the only platform-specific code.** `metal/` wraps
[rwkv-metal](https://github.com/RafaelUI/rwkv-metal)'s WKV kernels and adds
its own GEMV/GEMM kernels for the quantized layouts; `cuda/` is meant to be a
separate implementation and is not written yet. Both are expected to consume the same `.rwkvq` files
produced by `calibration/` + `formats/writer.py` — no backend-specific
calibration logic.

**Two entry points, one underlying config type.** `api.py`'s `quantize()`
accepts either a `preset` string (`"reduction"` / `"compression"`, defined in
`presets.py`) for quick start, or an explicit `QuantConfig` for full
control. Both paths end up as the same `QuantConfig` object consumed by
`calibration/fake_quant.py` — presets are just pre-filled configs, not a
separate code path.
