"""Конвертер RWKV-7 `.pth` -> `gguf` (F16) для llama.cpp.

Нужен для честного пересчёта сравнения с llama.cpp: их сторона меряется
относительно СВОЕЙ полноточной базы, а базового gguf у нас не было --
лежал только `Q4_K_M` от 16.07, снятый на прежней (утёкшей) постановке.
Апстримного конвертера на машине нет: `build-tools/llama.cpp` содержит
обрезанную копию `convert_hf_to_gguf.py` (296 строк, четыре модели, слова
`rwkv` нет ни разу), а HF-репозитория для g1h может не существовать
вовсе. Поэтому конвертируем из СВОЕГО `.pth`.

Схема имён и KV взяты не из документации, а из существующего
`rwkv7-g1h-1.5b-Q4_K_M.gguf`, который llama.cpp уже читает, и сверены с
исходниками загрузчика `src/models/rwkv7.cpp`. Три места, где легко
ошибиться молча:

1. ОРИЕНТАЦИЯ. gguf хранит массив формы reversed(ne). Линейные веса
   `.pth` лежат как `[out, in]` и попадают в файл КАК ЕСТЬ. А восемь
   низкоранговых лежат как `[in, rank]` (или `[rank, in]`) и должны быть
   ТРАНСПОНИРОВАНЫ. Проверка: `ffn.key.weight [8192, 2048]` даёт ne
   `[2048, 8192]`, а `att.a1 [2048, 96]` -- ne `[2048, 96]`, то есть в
   файле лежит `a1.T`.
2. СЛОЙ 0 НЕ КАК ОСТАЛЬНЫЕ. `rwkv7.cpp:85-88` создаёт `v1`/`v2` нулевого
   слоя с рангом ICLR (96), а не value-residual (64), с комментарием
   "actually not used". В `.pth` там честные 64, поэтому блок 0
   ДОПОЛНЯЕТСЯ НУЛЯМИ до 96 -- иначе загрузка падает на несовпадении
   формы. Значения не важны, тензор не используется.
3. ПОРЯДОК LERP. Шесть векторов сливаются в `time_mix_lerp_fused`
   формы ne `[n_embd, 1, 1, 6]`, и порядок срезов в
   `rwkv7-base.cpp:57-64` -- строго `r, w, k, v, a, g`.

Типы повторяют референс: одномерные и `w1`/`w2` -- F32, остальные
низкоранговые -- F16, большие матрицы -- F16.

    python tools/convert_pth_to_gguf.py IN.pth OUT.gguf [REF.gguf]
"""
import os
import sys

import numpy as np
import torch
import gguf

LERP = ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
VEC = ("w0", "a0", "v0", "k_k", "k_a")
# .pth -> gguf, крупные матрицы, кладутся КАК ЕСТЬ
LINEAR = {
    "att.receptance.weight": "time_mix_receptance.weight",
    "att.key.weight": "time_mix_key.weight",
    "att.value.weight": "time_mix_value.weight",
    "att.output.weight": "time_mix_output.weight",
    "ffn.key.weight": "channel_mix_key.weight",
    "ffn.value.weight": "channel_mix_value.weight",
}
# низкоранговые, кладутся ТРАНСПОНИРОВАННЫМИ
LORA = ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")
LORA_F32 = ("w1", "w2")


def f32(t) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()


def f16(t) -> np.ndarray:
    a = f32(t)
    bad = int(np.sum(~np.isfinite(a))) + int(np.sum(np.abs(a) > 65504.0))
    if bad:
        # Молча обрезать в inf нельзя: это тихая порча базы, относительно
        # которой считается Δppl всей их стороны.
        raise ValueError("значения вне диапазона F16: %d" % bad)
    return a.astype(np.float16)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser(
        "~/Develop/WKV-kvant/rwkv7-g1h-1.5b-Q4_K_M.gguf")

    print("читаю", src, flush=True)
    sd = torch.load(src, map_location="cpu", mmap=True, weights_only=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    n_embd = sd["emb.weight"].shape[1]
    n_vocab = sd["emb.weight"].shape[0]
    n_ff = sd["blocks.0.ffn.key.weight"].shape[0]
    head_size = sd["blocks.0.att.r_k"].shape[1]
    r_decay = sd["blocks.0.att.w1"].shape[1]
    r_iclr = sd["blocks.0.att.a1"].shape[1]
    r_gate = sd["blocks.0.att.g1"].shape[1]
    # ВАЖНО: ранг value-residual берётся из БЛОКА 1. В блоке 0 он тот же
    # по данным, но llama.cpp ждёт там ранг ICLR (см. докстринг).
    r_vres = sd["blocks.1.att.v1"].shape[1]
    print("L=%d D=%d vocab=%d ff=%d hs=%d ранги w/a/v/g=%d/%d/%d/%d"
          % (n_layer, n_embd, n_vocab, n_ff, head_size,
             r_decay, r_iclr, r_vres, r_gate), flush=True)

    print("беру токенизатор и chat_template из", ref, flush=True)
    rd = gguf.GGUFReader(ref)

    def kv(name):
        return rd.fields[name].contents()

    w = gguf.GGUFWriter(dst, "rwkv7")
    w.add_name(os.path.basename(src))
    w.add_context_length(int(kv("rwkv7.context_length")))
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(0)
    w.add_layer_norm_eps(float(kv("rwkv7.attention.layer_norm_epsilon")))
    w.add_wkv_head_size(head_size)
    w.add_decay_lora_rank(r_decay)
    w.add_iclr_lora_rank(r_iclr)
    w.add_value_residual_mix_lora_rank(r_vres)
    w.add_gate_lora_rank(r_gate)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    w.add_tokenizer_model("rwkv")
    w.add_token_list(kv("tokenizer.ggml.tokens"))
    w.add_token_types(kv("tokenizer.ggml.token_type"))
    w.add_bos_token_id(int(kv("tokenizer.ggml.bos_token_id")))
    w.add_eos_token_id(int(kv("tokenizer.ggml.eos_token_id")))
    w.add_eot_token_id(int(kv("tokenizer.ggml.eot_token_id")))
    w.add_chat_template(kv("tokenizer.chat_template"))

    # --- глобальные ---
    w.add_tensor("token_embd.weight", f16(sd["emb.weight"]))
    w.add_tensor("output.weight", f16(sd["head.weight"]))
    w.add_tensor("output_norm.weight", f32(sd["ln_out.weight"]))
    w.add_tensor("output_norm.bias", f32(sd["ln_out.bias"]))
    w.add_tensor("token_embd_norm.weight", f32(sd["blocks.0.ln0.weight"]))
    w.add_tensor("token_embd_norm.bias", f32(sd["blocks.0.ln0.bias"]))

    n_written = 6
    for i in range(n_layer):
        p = "blocks.%d." % i
        g = "blk.%d." % i

        w.add_tensor(g + "attn_norm.weight", f32(sd[p + "ln1.weight"]))
        w.add_tensor(g + "attn_norm.bias", f32(sd[p + "ln1.bias"]))
        w.add_tensor(g + "attn_norm_2.weight", f32(sd[p + "ln2.weight"]))
        w.add_tensor(g + "attn_norm_2.bias", f32(sd[p + "ln2.bias"]))
        w.add_tensor(g + "time_mix_ln.weight", f32(sd[p + "att.ln_x.weight"]))
        w.add_tensor(g + "time_mix_ln.bias", f32(sd[p + "att.ln_x.bias"]))
        n_written += 6

        # порядок r, w, k, v, a, g -- см. rwkv7-base.cpp:57-64
        fused = np.stack([f32(sd[p + "att." + n]).reshape(1, 1, n_embd)
                          for n in LERP], axis=0)
        w.add_tensor(g + "time_mix_lerp_fused.weight", fused)
        w.add_tensor(g + "channel_mix_lerp_k.weight",
                     f32(sd[p + "ffn.x_k"]).reshape(n_embd))
        n_written += 2

        for n in VEC:
            w.add_tensor(g + "time_mix_%s.weight" % n,
                         f32(sd[p + "att." + n]).reshape(n_embd))
        w.add_tensor(g + "time_mix_r_k.weight",
                     f32(sd[p + "att.r_k"]).reshape(n_embd))
        n_written += len(VEC) + 1

        for n in LORA:
            a = f32(sd[p + "att." + n]).T.copy()      # см. пункт 1 докстринга
            if i == 0 and n in ("v1", "v2"):          # см. пункт 2
                pad = np.zeros((r_iclr, n_embd) if n == "v1"
                               else (n_embd, r_iclr), dtype=np.float32)
                if n == "v1":
                    pad[:r_vres, :] = a
                else:
                    pad[:, :r_vres] = a
                a = pad
            w.add_tensor(g + "time_mix_%s.weight" % n,
                         a if n in LORA_F32 else a.astype(np.float16))
        n_written += len(LORA)

        for k, name in LINEAR.items():
            w.add_tensor(g + name, f16(sd[p + k]))
        n_written += len(LINEAR)

        print("  блок %d/%d" % (i + 1, n_layer), flush=True)

    print("тензоров записано:", n_written, flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("готово:", dst, "%.1f МБ" % (os.path.getsize(dst) / 1048576), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
