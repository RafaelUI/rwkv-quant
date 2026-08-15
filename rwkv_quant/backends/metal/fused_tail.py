"""Хвост TMix одним кернелем: group_norm + bonus + gate.

ЗАЧЕМ. После WKV идёт цепочка из примерно четырнадцати примитивов на
слой -- две редукции group_norm, вычитание, квадрат, sqrt, деление, вес,
смещение, три умножения bonus, редукция bonus, сложение и гейт, -- и
каждый из них работает над [H, S] = [32, 64] на декоде, то есть над
восемью килобайтами. На 24 слоя это порядка 336 запусков за токен при
объёме работы, который весь помещается в один. Разложение вычитанием
(`bench_step_decompose`) говорит, что вся не-GEMV часть шага стоит
4.0-5.0 мс при почти нулевом трафике, а трасса -- что шаг упирается в
ВЫДАЧУ инструкций (Instruction Throughput Limiter 92%) при ALU 2-17%.
То есть платим мы за запуски, а не за арифметику, и лечится это их
сокращением.

ПОРЯДОК СУММИРОВАНИЯ ДРУГОЙ, И ЭТО НЕ ПРИДИРКА. Обе редукции здесь
simd-древесные, а в MLX они свои; бит-в-бит требовать нельзя, и гейт
`tests/test_fuse_parity.py` для того и существует. Формулы при этом
взяты из `quant_model._group_norm` буквально, включая два прохода по
голове ради `mean((x-mean)^2)`: однопроходное `E[x^2] - mean^2`
арифметически то же, а численно -- разность близких величин, и на
нормировке это ровно то место, где она вылезает.

Кернель обслуживает произвольное N = B*T (одна threadgroup на пару
(строка, голова)), но включается только на decode-пути: на префилле
тензоры крупные, накладные запуска амортизируются, а лишняя реализация
на горячем пути стоит дороже, чем экономит (закон 23).
"""
import mlx.core as mx

_tail_cache = {}


def _get_kernel(H, S, eps):
    key = (H, S, eps)
    if key in _tail_cache:
        return _tail_cache[key]
    assert S % 32 == 0, f"хвост: S={S} не кратен ширине simdgroup"
    header = f"""
constant uint  H_C = {H};
constant uint  S_C = {S};
constant float EPS = {eps!r}f;
"""
    body = """
    uint tg   = threadgroup_position_in_grid.x;   // строка*H + голова
    uint lane = thread_position_in_threadgroup.x;
    uint h    = tg % H_C;
    uint base = tg * S_C;                         // [N*H, S] -- плотно
    uint hb   = h * S_C;                          // r_k / ln_x -- [H, S]

    // --- group_norm, два прохода: см. примечание в докстринге модуля
    float s1 = 0.0f;
    for (uint i = lane; i < S_C; i += 32) s1 += wkv[base + i];
    s1 = simd_sum(s1);
    float mean = s1 / (float)S_C;

    float s2 = 0.0f;
    for (uint i = lane; i < S_C; i += 32) {
        float d = wkv[base + i] - mean;
        s2 += d * d;
    }
    s2 = simd_sum(s2);
    float inv = 1.0f / sqrt(s2 / (float)S_C + EPS);

    // --- bonus: (r * k * r_k) по голове, затем на v
    float dp = 0.0f;
    for (uint i = lane; i < S_C; i += 32)
        dp += r[base + i] * k[base + i] * rk[hb + i];
    dp = simd_sum(dp);

    // --- сборка ровно в том же порядке, что и в нефьюзнутом пути:
    //     ((norm * w + b) + dp * v) * g
    for (uint i = lane; i < S_C; i += 32) {
        float nrm = (wkv[base + i] - mean) * inv;
        float y   = nrm * lnw[hb + i] + lnb[hb + i] + dp * v[base + i];
        out[base + i] = y * g[base + i];
    }
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv_tail_h{H}_s{S}",
        input_names=["wkv", "r", "k", "v", "rk", "lnw", "lnb", "g"],
        output_names=["out"],
        header=header, source=body,
    )
    _tail_cache[key] = kern
    return kern


def can_fuse_tail(H, S):
    return S % 32 == 0


def wkv_tail(wkv, r, k, v, r_k, ln_w, ln_b, g, H, S, eps=64e-5):
    """(group_norm(wkv) + bonus(r,k,r_k)*v) * g одним запуском.

    Все входы -- fp32 и плотные, любой формы с суммарным размером N*H*S
    (wkv/r/v приходят как [B, T, H, S], g как [B, T, D] -- это одна и та
    же память, кернелю нужен только плоский вид [N, D]).
    r_k/ln_w/ln_b -- [H*S]. Возвращает [N, D], то есть готовый вход
    o_proj; форму [B, T, D] восстанавливает вызывающий.
    """
    D = H * S
    n = wkv.size // D
    kern = _get_kernel(H, S, eps)
    return kern(
        inputs=[wkv.reshape(n, D), r.reshape(n, D), k.reshape(n, D),
                v.reshape(n, D), r_k.reshape(D), ln_w.reshape(D),
                ln_b.reshape(D), g.reshape(n, D)],
        grid=(n * H * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n, D)], output_dtypes=[mx.float32],
    )[0]
