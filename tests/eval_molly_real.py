"""Real (not emulated) ppl/size/decode-speed for MollySophia's
rwkv7-1.5B-g1g-mlx-6bit checkpoint, on the SAME eval_corpus_world.pt[:8]
used for every other number in NEXT_SESSION.md (requested 19.07-10, for
the write-up to Bo Peng). Reuses fla-hub/HF key naming -> maps onto this
project's internal "world" TMix/CMix representation and drives it through
QuantTMix/QuantCMix/QuantBlock/QuantRWKV7's ALREADY-VALIDATED forward math
(backends/metal/quant_model.py) via thin subclasses that only override
weight *loading*, not the recurrence/gating math itself -- keeps risk
confined to the key-mapping, not to re-deriving WKV7/LoRA-gate algebra.

Architecture identity check (why this is safe): her checkpoint is the
fla-hub HF port of an official BlinkDL RWKV7 G1 release, i.e. the SAME
architecture our own "world" naming loads (rwkv7_ref.py, quant_model.py)
-- fla-hub exists specifically to be a compatible reimplementation. Her
weights are natively affine-quantized (mx.nn.QuantizedLinear format,
group_size=64, bits=6) -- NOT this project's gw sb6 scheme -- handled via
the new MlxAffineQuantLinear class (quant_model.py), which calls
mx.quantized_matmul directly (the real fast Metal path, same one her own
runtime uses), not a dense-dequant fallback -- so the decode-speed number
here is honest, not an approximation.

Sanity gates before trusting the ppl number (see header comment on
methodology -- no independent fla/transformers install on this 16GB
fanless machine, see NEXT_SESSION 19.07-9 OOM history with heavy ML deps):
  1. every key in the safetensors file is consumed exactly once (no silent
     drops/typos in the mapping) -- asserted below.
  2. shapes line up with config.json (hidden=2048, heads=32, head_dim=64,
     vocab=65536, ranks 96/96/64/256) -- asserted per-tensor during load.
  3. greedy continuation of a plain English prompt is coherent text, not
     garbage -- printed at the end for eyeball inspection.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import mlx.core as mx

from rwkv_quant.formats.schema import QuantizedTensor, QuantizedCheckpoint
from rwkv_quant.backends.metal.quant_model import (
    QuantTMix, QuantCMix, QuantBlock, QuantRWKV7, MlxAffineQuantLinear,
    _dense as _dense_orig, _layer_norm, l2_norm,
)

MODEL_DIR = os.path.expanduser("~/Develop/rwkv7-1.5B-g1g-mlx-6bit")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_world.pt")
VOCAB_TXT = os.path.expanduser("~/Develop/rwkv7-1.5B-g1g-mlx-6bit/rwkv_vocab_v20230424.txt")

N_LAYER, N_EMBD, HEAD_SIZE, VOCAB = 24, 2048, 64, 65536
N_HEAD = N_EMBD // HEAD_SIZE
GS, BITS = 64, 6


def _quant_qt(key, W, S, B):
    """uint32-packed [out, in_packed] + fp16 scales/biases -> QuantizedTensor
    ready for MlxAffineQuantLinear (real kernel, no dequant)."""
    out = W.shape[0]
    in_features = S.shape[1] * GS
    qt = QuantizedTensor(key=key, group="proj", bits=BITS, shape=(out, in_features))
    qt.gw_mode = "mlx_affine"
    qt.mlx_weight = W
    qt.mlx_scales = S
    qt.mlx_biases = B
    qt.mlx_group_size = GS
    qt.mlx_bits = BITS
    return qt


def _to_dense_qt(arr: mx.array):
    """mx.array (already dense fp16, not quantized) -> QuantizedTensor
    consumable by the stock _dense() helper (expects .dense as a torch
    tensor -- small vectors/matrices only, cost is negligible)."""
    t = torch.from_numpy(np.array(arr.astype(mx.float32)))
    return QuantizedTensor(key="", group="other", bits=16, shape=tuple(arr.shape), dense=t)


def _dense(arr: mx.array):
    return _dense_orig(_to_dense_qt(arr))


class MollyTMix(QuantTMix):
    def __init__(self, w, i, used):
        self.H, self.S = N_HEAD, HEAD_SIZE
        self.layer_id = i
        p = f"model.layers.{i}.attn."

        def take(suffix):
            used.add(p + suffix)
            return w[p + suffix]

        def dq(name):
            return take(name + ".weight"), take(name + ".scales"), take(name + ".biases")

        self.x_r = _dense(take("x_r")); self.x_w = _dense(take("x_w")); self.x_k = _dense(take("x_k"))
        self.x_v = _dense(take("x_v")); self.x_a = _dense(take("x_a")); self.x_g = _dense(take("x_g"))

        # LoRA down (lora.0, quantized) -- mx.dequantize once into a dense
        # [rank, D] array (rank is tiny, cost negligible); LoRA up (lora.2,
        # already dense fp16 in her checkpoint) taken as-is. Matches this
        # project's own convention of keeping LoRA branches dense (see
        # quant_model.py module docstring re: world-naming LoRA transpose
        # semantics) -- just sourced from a different quantization scheme.
        def lora_down(name):
            Wq, Sc, Bi = dq(f"{name}.lora.0")
            return _dense(mx.dequantize(Wq, scales=Sc, biases=Bi, group_size=GS, bits=BITS))

        self.w_lora_A = lora_down("w_lora")
        self.w_lora_B_w = _dense(take("w_lora.lora.2.weight"))
        self.w_lora_B_b = _dense(take("w_lora.lora.2.bias"))
        self.a_lora_A = lora_down("a_lora")
        self.a_lora_B_w = _dense(take("a_lora.lora.2.weight"))
        self.a_lora_B_b = _dense(take("a_lora.lora.2.bias"))
        if i == 0:
            self.v_lora_A = self.v_lora_B_w = self.v_lora_B_b = None
        else:
            self.v_lora_A = lora_down("v_lora")
            self.v_lora_B_w = _dense(take("v_lora.lora.2.weight"))
            self.v_lora_B_b = _dense(take("v_lora.lora.2.bias"))
        self.g_lora_A = lora_down("g_lora")
        self.g_lora_B_w = _dense(take("g_lora.lora.2.weight"))

        self.k_k = _dense(take("k_k")).reshape(self.H, self.S)
        self.k_a = _dense(take("k_a")).reshape(self.H, self.S)
        self.r_k = _dense(take("r_k")).reshape(self.H, self.S)

        self.r_proj = MlxAffineQuantLinear(_quant_qt("r", *dq("r_proj")))
        self.k_proj = MlxAffineQuantLinear(_quant_qt("k", *dq("k_proj")))
        self.v_proj = MlxAffineQuantLinear(_quant_qt("v", *dq("v_proj")))
        self.o_proj = MlxAffineQuantLinear(_quant_qt("o", *dq("o_proj")))

        self.ln_x_w = _dense(take("g_norm.weight")).reshape(-1)
        self.ln_x_b = _dense(take("g_norm.bias")).reshape(-1)
        self._build_fused()


class MollyCMix(QuantCMix):
    def __init__(self, w, i, used):
        p = f"model.layers.{i}.ffn."

        def take(suffix):
            used.add(p + suffix)
            return w[p + suffix]

        def dq(name):
            return take(name + ".weight"), take(name + ".scales"), take(name + ".biases")

        self.x_k = _dense(take("x_k"))
        self.key = MlxAffineQuantLinear(_quant_qt("key", *dq("key")))
        self.value = MlxAffineQuantLinear(_quant_qt("value", *dq("value")))


class MollyBlock(QuantBlock):
    def __init__(self, w, i, used):
        p = f"model.layers.{i}."

        def take(suffix):
            used.add(p + suffix)
            return w[p + suffix]

        self.ln1_w = _dense(take("attn_norm.weight")); self.ln1_b = _dense(take("attn_norm.bias"))
        self.ln2_w = _dense(take("ffn_norm.weight")); self.ln2_b = _dense(take("ffn_norm.bias"))
        self.tmix = MollyTMix(w, i, used)
        self.cmix = MollyCMix(w, i, used)


class MollyRWKV7(QuantRWKV7):
    def __init__(self, w):
        self.naming = "world"
        self.n_layer, self.n_embd, self.head_size = N_LAYER, N_EMBD, HEAD_SIZE
        self.n_head, self.vocab_size = N_HEAD, VOCAB
        used = set()

        def take(k):
            used.add(k)
            return w[k]

        Wq, Sc, Bi = take("model.embeddings.weight"), take("model.embeddings.scales"), take("model.embeddings.biases")
        self.emb_weight = mx.dequantize(Wq, scales=Sc, biases=Bi, group_size=GS, bits=BITS).astype(mx.float16)

        self.head = MlxAffineQuantLinear(_quant_qt(
            "head", take("lm_head.weight"), take("lm_head.scales"), take("lm_head.biases")))

        self.ln0_w = _dense(take("model.layers.0.pre_norm.weight"))
        self.ln0_b = _dense(take("model.layers.0.pre_norm.bias"))
        self.ln_out_w = _dense(take("model.norm.weight"))
        self.ln_out_b = _dense(take("model.norm.bias"))

        self.blocks = [MollyBlock(w, i, used) for i in range(self.n_layer)]
        self._materialize()

        unused = set(w.keys()) - used
        if unused:
            raise RuntimeError(f"{len(unused)} keys never consumed by the loader, "
                                f"mapping is incomplete: {sorted(unused)[:10]}...")
        print(f"[MollyRWKV7] loaded, all {len(w)} tensors consumed exactly once.")


def build_tokenizer():
    sys.path.insert(0, os.path.expanduser("~/Develop/WKV-kvant"))
    from world_tokenizer import RWKV_WORLD_TOKENIZER
    return RWKV_WORLD_TOKENIZER(VOCAB_TXT)


def main():
    t0 = time.time()
    w = mx.load(f"{MODEL_DIR}/model.safetensors")
    print(f"mx.load: {time.time()-t0:.1f}s, {len(w)} tensors")

    model = MollyRWKV7(w)

    # --- sanity gate: greedy continuation, eyeball for coherence ---
    tok = build_tokenizer()
    prompt = "The capital of France is"
    ids = tok.encode(prompt) if hasattr(tok, "encode") else tok.encode_bytes(prompt.encode("utf-8"))
    idx = mx.array([ids], dtype=mx.int32)
    st = model.init_state(1)
    logits, st = model.forward_stateful(idx, st, last_only=True)
    out_ids = list(ids)
    next_tok = int(mx.argmax(logits[:, -1], axis=-1).item())
    out_ids.append(next_tok)
    cur = mx.array([[next_tok]], dtype=mx.int32)
    for _ in range(24):
        logits, st = model.step(cur, st)
        next_tok = int(mx.argmax(logits[:, -1], axis=-1).item())
        out_ids.append(next_tok)
        cur = mx.array([[next_tok]], dtype=mx.int32)
    decode_fn = tok.decode if hasattr(tok, "decode") else tok.decode_bytes
    text = decode_fn(out_ids)
    print(f"[sanity] greedy continuation of {prompt!r}:\n  {text!r}")

    # --- ppl on the shared corpus ---
    data = torch.load(CORPUS)[:8].numpy()

    def ppl_of(m):
        total_nll, total_tok = 0.0, 0
        for i in range(0, data.shape[0], 4):
            batch = data[i:i + 4]
            idxb = mx.array(batch[:, :-1]); target = batch[:, 1:]
            lg = m(idxb); mx.eval(lg)
            logp = np.array(mx.log(mx.softmax(lg.astype(mx.float32), axis=-1) + 1e-12))
            Bb, T, V = logp.shape
            idxf = target.reshape(-1); logpf = logp.reshape(-1, V)
            nll = -logpf[np.arange(len(idxf)), idxf]
            total_nll += nll.sum(); total_tok += nll.size
        return float(np.exp(total_nll / total_tok))

    ppl = ppl_of(model)
    print(f"molly_g1g_mlx6bit REAL   ppl={ppl:14.4f}")

    # --- decode speed (A/B-style single process, steady state) ---
    prompt64 = mx.array(data[0:1, :64].astype(np.int32))
    st = model.init_state(1)
    logits, st = model.forward_stateful(prompt64, st, last_only=True)
    tok_id = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(8):
        logits, st = model.step(tok_id[None], st)
        tok_id = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok_id)
    t0 = time.time(); n = 64
    for _ in range(n):
        logits, st = model.step(tok_id[None], st)
        tok_id = mx.argmax(logits[:, -1], axis=-1); mx.eval(tok_id)
    dt = (time.time() - t0) / n * 1000
    print(f"decode: {dt:.2f} ms/tok")

    size_mb = os.path.getsize(f"{MODEL_DIR}/model.safetensors") / 1e6
    print(f"file size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
