"""Открытый вопрос №1 из NEXT_SESSION.md: разрыв real-quant vs fake_quant на
world-naming LoRA (~20x по ppl), необъяснённый в предыдущей сессии.

Гипотеза (новая, не проверенная старой сессией): дело не в ГРАНУЛЯРНОСТИ
(2048 групп у writer.py vs 96 групп у RWKV7Ref), а в ТОМ, ПО КАКОЙ ОСИ
идёт группировка относительно оси свёртки (contraction axis) в матмуле.

Стандартная RTN-конвенция для nn.Linear-веса [out,in]: per-row = per-output-
channel, ось свёртки (in) остаётся ВНУТРИ группы. Тогда scale можно вынести
за скобки суммы: y_j = scale_j * sum_i(x_i * code_ji) -- ошибка квантования
не корродирует сумму систематически по разным j.

writer.py квантует СЫРОЙ world-тензор w1 формы [C, rank] = [in, out] (ДО
транспозиции). "per-row" там = per-input-channel (C), то есть группировка
идёт ВДОЛЬ оси свёртки в терминах эффективного матмула x @ w1 (contraction
над C). Тогда per-input scale_i НЕЛЬЗЯ вынести за сумму: y_j = sum_i(x_i *
scale_i * code_ij) -- ошибки по разным i складываются некогерентно внутри
каждого выходного j. Это не "грубее/точнее", это другая по сути ошибка,
которая не отражается в простом reconstruction error (|W - W_hat|), но
отражается в error матмула (|x@W - x@W_hat|).

Проверяем: (1) сырой reconstruction error поэлементно в обеих ориентациях;
(2) downstream matmul error на случайном x -- где, по гипотезе, должен быть
на порядок хуже для writer.py-ориентации (raw, per-input-row) чем для
RWKV7Ref-ориентации (transposed, per-output-row), несмотря на то что у
writer.py групп в 21x больше (2048 vs 96)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from rwkv_quant.formats.writer import _real_quantize

CKPT_PTH = os.path.expanduser("~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth")
BITS = 4
LAYER = 1  # не 0: v1/v2/v0 отсутствуют в layer 0 (см. rwkv7_ref.py: "unused at layer 0")
KEYS = ["w1", "a1", "v1", "g1"]  # LoRA-A матрицы, форма [C, rank] сырая


def dequant(codes, scale):
    return codes.float() * scale.float()


def main():
    torch.manual_seed(0)
    sd = torch.load(CKPT_PTH, map_location="cpu")

    print(f"{'tensor':<6} {'raw_shape':<12} "
          f"{'recon_err(raw)':<16} {'recon_err(T)':<14} "
          f"{'matmul_err(raw)':<17} {'matmul_err(T)':<14} {'ratio':<8}")

    for key in KEYS:
        full_key = f"blocks.{LAYER}.att.{key}"
        if full_key not in sd:
            print(f"{key}: KEY NOT FOUND ({full_key}), skipping")
            continue
        w_raw = sd[full_key].float()          # [C, rank] = [in, out], как хранится на диске
        w_T = w_raw.T.contiguous()             # [rank, C] = [out, in], как использует RWKV7Ref

        # --- (1) сырая RTN-квантизация в обеих ориентациях ---
        codes_raw, scale_raw = _real_quantize(w_raw, BITS)
        w_raw_hat = dequant(codes_raw, scale_raw)
        recon_err_raw = (w_raw - w_raw_hat).abs().mean().item() / w_raw.abs().mean().item()

        codes_T, scale_T = _real_quantize(w_T, BITS)
        w_T_hat = dequant(codes_T, scale_T)
        recon_err_T = (w_T - w_T_hat).abs().mean().item() / w_T.abs().mean().item()

        # --- (2) downstream matmul error: x @ W, свёртка по оси C (=in) ---
        # эффективный forward: xw[...,C] @ w1[C,rank] -> [...,rank]
        # (тот путь, которым w1 реально участвует в вычислении, что для
        # writer-ориентации (raw), что для RWKV7Ref-ориентации (transposed,
        # но потом обратно транспонированной в forward -- см. rwkv7_ref.py
        # строка ~173: q(t.w_lora_A, ...).T)
        x = torch.randn(1000, w_raw.shape[0])  # [N, C]

        y_true = x @ w_raw                                  # [N, rank], эталон

        y_raw_hat = x @ w_raw_hat                            # writer.py-путь: квантован сырой [C,rank]
        matmul_err_raw = (y_true - y_raw_hat).norm().item() / y_true.norm().item()

        y_T_hat = x @ w_T_hat.T                               # RWKV7Ref-путь: квантован [rank,C], потом .T обратно
        matmul_err_T = (y_true - y_T_hat).norm().item() / y_true.norm().item()

        ratio = matmul_err_raw / max(matmul_err_T, 1e-12)

        print(f"{key:<6} {str(tuple(w_raw.shape)):<12} "
              f"{recon_err_raw:<16.6f} {recon_err_T:<14.6f} "
              f"{matmul_err_raw:<17.6f} {matmul_err_T:<14.6f} {ratio:<8.2f}")


if __name__ == "__main__":
    main()


def branch_test():
    """Полная A->tanh->B(+bias) цепочка для ветки 'w' (decay-gate) --
    самая чувствительная ветка (управляет затуханием на КАЖДОМ шаге
    рекуррентности, ошибка потенциально накапливается мультипликативно
    по времени и слоям, в отличие от одиночного матмула из main())."""
    import torch.nn.functional as F
    print("\n--- full A->tanh->B(+bias) chain, branch 'w', layer", LAYER, "---")
    sd = torch.load(CKPT_PTH, map_location="cpu")
    p = f"blocks.{LAYER}.att."
    w1 = sd[p + "w1"].float()   # [C, rank] raw
    w2 = sd[p + "w2"].float()   # [rank, C] raw
    w0 = sd[p + "w0"].float().reshape(-1)  # [C], всегда dense (уже пофикшено)

    x = torch.randn(1000, w1.shape[0])  # [N, C], синтетический вход

    def forward_raw(w1_, w2_):
        # writer.py-путь: A и B используются В СЫРОЙ ориентации, как их видит
        # quantize_tensor() -- т.е. как если бы отдельный кернел знал про
        # world-специфичную (in,out) раскладку и работал без транспонирования
        h = torch.tanh(x @ w1_)          # [N, rank]
        return h @ w2_ + w0              # [N, C]

    def forward_ref(w1_, w2_):
        # RWKV7Ref-путь: A и B транспонируются ПОСЛЕ загрузки/деквантования,
        # используются как стандартный nn.Linear [out,in]
        w1T = w1_.T.contiguous()          # [rank, C] = [out,in] для A-матмула h=x@w1.T.T? см. rwkv7_ref.py
        w2T = w2_.T.contiguous()          # [C, rank] = [out,in] для B (F.linear)
        h = torch.tanh(F.linear(x, w1T))  # x[N,C] @ w1T.T[C,rank] -> [N,rank]; w1T=[rank,C] значит w1T.T=[C,rank]=w1_ (эквивалентно x@w1_)
        return F.linear(h, w2T, w0)       # h[N,rank] @ w2T.T[rank,C] -> [N,C]; w2T=[C,rank] раз w2T.T=[rank,C]=w2_ (эквивалентно h@w2_)

    y_true = forward_raw(w1, w2)  # эталон (обе формулы математически идентичны на fp32 весах)
    assert (y_true - forward_ref(w1, w2)).abs().max().item() < 1e-3, "sanity: two formulas should agree in fp32"

    # --- writer.py путь: квантуем СЫРЫЕ w1,w2 (как это реально делает quantize_tensor) ---
    c1, s1 = _real_quantize(w1, BITS); w1_hat_raw = dequant(c1, s1)
    c2, s2 = _real_quantize(w2, BITS); w2_hat_raw = dequant(c2, s2)
    y_writer = forward_raw(w1_hat_raw, w2_hat_raw)

    # --- RWKV7Ref/fake_quant путь: квантуем ТРАНСПОНИРОВАННЫЕ w1,w2 ---
    c1t, s1t = _real_quantize(w1.T.contiguous(), BITS); w1_hat_T = dequant(c1t, s1t).T.contiguous()
    c2t, s2t = _real_quantize(w2.T.contiguous(), BITS); w2_hat_T = dequant(c2t, s2t).T.contiguous()
    y_ref = forward_raw(w1_hat_T, w2_hat_T)  # та же формула forward_raw, но веса деквантованы из T-ориентации

    err_writer = (y_true - y_writer).norm().item() / y_true.norm().item()
    err_ref = (y_true - y_ref).norm().item() / y_true.norm().item()
    print(f"decay-branch output rel err: writer.py-orientation={err_writer:.6f}  "
          f"RWKV7Ref-orientation={err_ref:.6f}  ratio={err_writer/max(err_ref,1e-12):.3f}")

    # --- усиление через decay: w_ = -softplus(-h) - 0.5, затем экспонента на T шагов ---
    def decay_from(y):
        w_ = -F.softplus(-y) - 0.5
        return torch.exp(w_)  # decay multiplier per step

    d_true = decay_from(y_true)
    d_writer = decay_from(y_writer)
    d_ref = decay_from(y_ref)
    for T in (1, 32, 256, 1024):
        amp_writer = (d_true ** T - d_writer ** T).abs().mean().item()
        amp_ref = (d_true ** T - d_ref ** T).abs().mean().item()
        print(f"  after T={T:>5} decay steps: |Δdecay^T| writer={amp_writer:.6e}  ref={amp_ref:.6e}  "
              f"ratio={amp_writer/max(amp_ref,1e-30):.2f}")


if __name__ == "__main__":
    branch_test()


def g_branch_test():
    """Ветка 'g' устроена иначе, чем 'w': sigmoid стоит МЕЖДУ A и B
    (g = sigmoid(x@A) @ B), а не после B с bias, как у w/a/v. Значит выход
    g НИЧЕМ не ограничен сверху -- в отличие от w/a/v, где финальная
    sigmoid/softplus зажимает любую ошибку квантования в фиксированный
    диапазон. Проверяем: даёт ли ориентация writer.py (сырая, per-input-row
    для B) катастрофически иную ошибку именно здесь, в отличие от w-ветки,
    где обе ориентации были практически равноценны."""
    print("\n--- g-branch (sigmoid BEFORE B, no final bound), layer", LAYER, "---")
    sd = torch.load(CKPT_PTH, map_location="cpu")
    p = f"blocks.{LAYER}.att."
    g1 = sd[p + "g1"].float()   # [C, rank] raw = [2048, 256]
    g2 = sd[p + "g2"].float()   # [rank, C] raw = [256, 2048]

    x = torch.randn(1000, g1.shape[0])

    def forward_raw(g1_, g2_):
        h = torch.sigmoid(x @ g1_)   # [N, rank] -- ограничен sigmoid уже здесь
        return h @ g2_               # [N, C] -- НИЧЕМ не ограничен после этого

    y_true = forward_raw(g1, g2)

    # writer.py путь: квантуем сырые g1 (per-row over C=2048), g2 (per-row over rank=256)
    c1, s1 = _real_quantize(g1, BITS); g1_hat_raw = dequant(c1, s1)
    c2, s2 = _real_quantize(g2, BITS); g2_hat_raw = dequant(c2, s2)
    y_writer = forward_raw(g1_hat_raw, g2_hat_raw)

    # RWKV7Ref путь: квантуем транспонированные (per-row over rank для A, per-row over C для B)
    c1t, s1t = _real_quantize(g1.T.contiguous(), BITS); g1_hat_T = dequant(c1t, s1t).T.contiguous()
    c2t, s2t = _real_quantize(g2.T.contiguous(), BITS); g2_hat_T = dequant(c2t, s2t).T.contiguous()
    y_ref = forward_raw(g1_hat_T, g2_hat_T)

    err_writer = (y_true - y_writer).norm().item() / y_true.norm().item()
    err_ref = (y_true - y_ref).norm().item() / y_true.norm().item()
    max_writer = (y_true - y_writer).abs().max().item()
    max_ref = (y_true - y_ref).abs().max().item()
    print(f"g-branch output rel err: writer={err_writer:.6f}  ref={err_ref:.6f}  ratio={err_writer/max(err_ref,1e-12):.3f}")
    print(f"g-branch output MAX abs err: writer={max_writer:.6f}  ref={max_ref:.6f}  ratio={max_writer/max(max_ref,1e-12):.3f}")

    # g2-only isolation (g1 exact, только g2 квантован) -- сопоставимо с реальным диагнозом g_only_g2
    y_writer_g2only = forward_raw(g1, g2_hat_raw)
    y_ref_g2only = forward_raw(g1, g2_hat_T)
    err_w2 = (y_true - y_writer_g2only).norm().item() / y_true.norm().item()
    err_r2 = (y_true - y_ref_g2only).norm().item() / y_true.norm().item()
    max_w2 = (y_true - y_writer_g2only).abs().max().item()
    max_r2 = (y_true - y_ref_g2only).abs().max().item()
    print(f"g2-ONLY rel err: writer={err_w2:.6f}  ref={err_r2:.6f}  ratio={err_w2/max(err_r2,1e-12):.3f}")
    print(f"g2-ONLY MAX abs err: writer={max_w2:.6f}  ref={max_r2:.6f}  ratio={max_w2/max(max_r2,1e-12):.3f}")


if __name__ == "__main__":
    g_branch_test()
