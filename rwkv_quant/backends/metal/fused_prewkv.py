"""Пред-WKV блок одной функцией: w, a, kk, обновление k и v.

ЗАЧЕМ. Между GEMV r/k/v и самим сканом идёт около пяти запусков на слой
(sigmoid, exp, l2_norm с редукцией, две поправки), и на декоде это
0.387 мс из 13.2 при дне по памяти около нуля -- то есть чистый оверхед
выдачи, ровно как хвост TMix. Кернель бьёт в эту статью.

ОДНА РЕАЛИЗАЦИЯ НА ОБА ПУТИ. Формулы стояли инлайном дважды -- в
forward_stateful и в _forward_stateful_fused, -- и различались только
порядком объявления w и a. Здесь они сведены в одну функцию: чтобы
правка не переезжала между копиями вручную (закон 23) и чтобы у гейта
ядра был эталон, которым пользуется САМА МОДЕЛЬ, а не его копия внутри
теста (закон 30).

prewkv_ref БИТ-В-БИТ повторяет прежний инлайн: порядок операций и типы
не тронуты, вынос -- чистое перемещение кода.
"""
import mlx.core as mx


def l2_norm(x):
    return x / mx.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-12)


def prewkv_ref(y_w, y_a, y_v, k, v, v_first, w_b, a_b, v_b,
               k_k, k_a, layer_id, B, T, H, S, dtype):
    """Возвращает (w, k, v, -kk, kk*a, v_first) -- вход WKV-скана.

    y_w, y_a, y_v -- выходы LoRA ПОСЛЕ up-проекции и ДО прибавления
    bias, форма [B, T, D]; y_v = None на нулевом слое. k, v -- [B, T,
    H, S] с выхода проекций. k_k, k_a -- [H, S]. dtype -- рабочий тип
    пути (x.dtype): w считается через fp32 и приводится обратно.
    """
    w = y_w + w_b
    w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(dtype)
    w = w.reshape(B, T, H, S)
    a = mx.sigmoid(y_a + a_b).reshape(B, T, H, S)

    kk = l2_norm(k * k_k)
    k = k * (1.0 + (a - 1.0) * k_a)

    if layer_id == 0:
        v_first = v
    else:
        vv = mx.sigmoid(y_v + v_b).reshape(B, T, H, S)
        v = v + (v_first - v) * vv

    return w, k, v, -kk, kk * a, v_first




# ---------------------------------------------------------------------------
# ЯДРО
#
# КОНСТРУКЦИЯ ПРОДИКТОВАНА ЗАМЕРОМ (tests/_sess/probe_mlx_bitmatch.py),
# а не удобством. Три числа, которые её задали:
#
# 1. Metal СВОРАЧИВАЕТ a*b+c в fma. Цепочка k*(1+(a-1)*k_a), написанная
#    одним выражением, совпадает с MLX побитово лишь на 49.7% и врёт до
#    410 ulp; явная fma даёт РОВНО ТО ЖЕ, то есть свёртка подтверждена
#    прямо. Та же цепочка, разложенная на именованные промежуточные, --
#    100.00% бит-в-бит. Поэтому здесь всё пишется по шагам, и слитные
#    выражения в этом файле -- не стилистика, а поломка.
#
# 2. СОБСТВЕННЫЙ sigmoid MLX-овский НЕ ВОСПРОИЗВОДИТ: лучшие формулы
#    (прямая, precise::exp, precise::divide -- все три совпадают между
#    собой) дают 93.2% и до 2 ulp. Порог на k и v -- бит-в-бит, поэтому
#    sigmoid для a и vv считает MLX и отдаёт сюда ГОТОВЫМ. Это стоит
#    четырёх запусков снаружи и делает k и v чистой цепочкой.
#    Для w свой sigmoid допустим: там порог relmax 1e-6, а 2 ulp на
#    sigmoid плюс 1 на exp -- это 3.6e-7, запас втрое.
#
# 3. l2_norm накапливается ПОСЛЕДОВАТЕЛЬНО, а не simd-деревом: дерево
#    совпадает на 51.9% и до 4 ulp, последовательный проход -- 84.9% и
#    до 2 ulp. На 64 элементах цена -- 64 итерации в лейне, вдвое лучший
#    запас по порогу того стоит.
#
# Геометрия -- как у fused_tail: одна threadgroup на пару (строка,
# голова), 32 лейна. Сумма считается каждым лейном целиком: результат у
# всех один и тот же, зато нет ни simd_sum, ни барьера.
# ---------------------------------------------------------------------------
_cache = {}


def _get_kernel(H, S, layer0):
    key = (H, S, layer0)
    if key in _cache:
        return _cache[key]
    assert S % 32 == 0, "пред-WKV: S=%d не кратен ширине simdgroup" % S
    header = ("\nconstant uint H_C = %d;\nconstant uint S_C = %d;\n"
              "constant bool L0 = %s;\n" % (H, S, "true" if layer0 else "false"))
    body = """
    uint tg   = threadgroup_position_in_grid.x;   // строка*H + голова
    uint lane = thread_position_in_threadgroup.x;
    uint h    = tg % H_C;
    uint base = tg * S_C;                         // [N*H, S] -- плотно
    uint hb   = h * S_C;                          // k_k / k_a -- [H, S]

    // --- сумма квадратов для l2_norm: последовательно, по шагам
    float s2 = 0.0f;
    for (uint i = 0; i < S_C; ++i) {
        float t  = kin[base + i] * kk_[hb + i];
        float sq = t * t;
        s2 = s2 + sq;
    }
    float r = sqrt(s2 + 1e-12f);

    for (uint i = lane; i < S_C; i += 32) {
        uint j = base + i;

        // w = exp(-0.606531 * sigmoid(zw))
        float zwv = zw[j];
        float sw  = 1.0f / (1.0f + exp(-zwv));
        float ew  = -0.606531f * sw;
        wout[j]   = exp(ew);

        // kk = l2_norm(k * k_k), затем -kk и kk*a
        float t   = kin[j] * kk_[hb + i];
        float kkv = t / r;
        nkkout[j] = -kkv;
        float av  = sa[j];
        kkaout[j] = kkv * av;

        // k = k * (1 + (a - 1) * k_a) -- ТОЛЬКО ПО ШАГАМ, см. пункт 1
        float d   = av - 1.0f;
        float p   = d * ka_[hb + i];
        float u   = 1.0f + p;
        kout[j]   = kin[j] * u;

        // v = v + (v_first - v) * vv -- тоже по шагам
        float v0 = vin[j];
        if (L0) {
            vout[j] = v0;
        } else {
            float dv = vf[j] - v0;
            float tv = dv * sv[j];
            vout[j]  = v0 + tv;
        }
    }
"""
    kern = mx.fast.metal_kernel(
        name="prewkv_h%d_s%d_l%d" % (H, S, int(layer0)),
        input_names=["zw", "sa", "sv", "kin", "vin", "vf", "kk_", "ka_"],
        output_names=["wout", "kout", "vout", "nkkout", "kkaout"],
        header=header, source=body,
    )
    _cache[key] = kern
    return kern


def can_fuse_prewkv(H, S, n, dtype):
    """Ядро считает в fp32 и включается только на декоде: на префилле
    тензоры крупные, накладные запуска амортизируются."""
    return S % 32 == 0 and n == 1 and dtype == mx.float32


def prewkv_kernel(y_w, y_a, y_v, k, v, v_first, w_b, a_b, v_b,
                  k_k, k_a, layer_id, B, T, H, S, dtype):
    """Сигнатура та же, что у prewkv_ref: диспетчер зовёт их как одно."""
    D = H * S
    n = k.size // D
    l0 = (layer_id == 0)
    zw = (y_w + w_b).reshape(n, D)
    sa = mx.sigmoid(y_a + a_b).reshape(n, D)
    if l0:
        sv = zw                       # заглушка: под L0 не читается
        vf = v.reshape(n, D)
    else:
        sv = mx.sigmoid(y_v + v_b).reshape(n, D)
        vf = v_first.reshape(n, D)
    kern = _get_kernel(H, S, l0)
    w, ko, vo, nkk, kka = kern(
        inputs=[zw, sa, sv, k.reshape(n, D), v.reshape(n, D), vf,
                k_k.reshape(D), k_a.reshape(D)],
        grid=(n * H * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n, D)] * 5, output_dtypes=[mx.float32] * 5,
    )
    shp = (B, T, H, S)
    w, ko, vo = w.reshape(shp), ko.reshape(shp), vo.reshape(shp)
    nkk, kka = nkk.reshape(shp), kka.reshape(shp)
    return w, ko, vo, nkk, kka, (vo if l0 else v_first)


# --- ВАРИАНТ С sigmoid ВНУТРИ: только для замера ---------------------------
# Численно НЕ проходит порог бит-в-бит на k и v (свой sigmoid отстоит от
# MLX-овского на 2 ulp, замер в probe_mlx_bitmatch). Существует, чтобы
# разделить две причины нулевого выигрыша: дорога ли обвязка снаружи или
# дорого само ядро. В decode-путь не подключён.
_cache_in = {}


def _get_kernel_inside(H, S, layer0):
    key = (H, S, layer0)
    if key in _cache_in:
        return _cache_in[key]
    header = ("\nconstant uint H_C = %d;\nconstant uint S_C = %d;\n"
              "constant bool L0 = %s;\n" % (H, S, "true" if layer0 else "false"))
    body = """
    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint h    = tg % H_C;
    uint base = tg * S_C;
    uint hb   = h * S_C;

    float s2 = 0.0f;
    for (uint i = 0; i < S_C; ++i) {
        float t  = kin[base + i] * kk_[hb + i];
        float sq = t * t;
        s2 = s2 + sq;
    }
    float r = sqrt(s2 + 1e-12f);

    for (uint i = lane; i < S_C; i += 32) {
        uint j = base + i;
        float zwv = yw[j] + wb[hb + i];
        float sw  = 1.0f / (1.0f + exp(-zwv));
        float ew  = -0.606531f * sw;
        wout[j]   = exp(ew);

        float t   = kin[j] * kk_[hb + i];
        float kkv = t / r;
        nkkout[j] = -kkv;
        float zav = ya[j] + ab[hb + i];
        float av  = 1.0f / (1.0f + exp(-zav));
        kkaout[j] = kkv * av;

        float d   = av - 1.0f;
        float p   = d * ka_[hb + i];
        float u   = 1.0f + p;
        kout[j]   = kin[j] * u;

        float v0 = vin[j];
        if (L0) {
            vout[j] = v0;
        } else {
            float zvv = yv[j] + vb[hb + i];
            float sv2 = 1.0f / (1.0f + exp(-zvv));
            float dv  = vf[j] - v0;
            float tv  = dv * sv2;
            vout[j]   = v0 + tv;
        }
    }
"""
    kern = mx.fast.metal_kernel(
        name="prewkv_in_h%d_s%d_l%d" % (H, S, int(layer0)),
        input_names=["yw", "ya", "yv", "wb", "ab", "vb",
                     "kin", "vin", "vf", "kk_", "ka_"],
        output_names=["wout", "kout", "vout", "nkkout", "kkaout"],
        header=header, source=body,
    )
    _cache_in[key] = kern
    return kern


def prewkv_kernel_inside(y_w, y_a, y_v, k, v, v_first, w_b, a_b, v_b,
                         k_k, k_a, layer_id, B, T, H, S, dtype):
    D = H * S
    n = k.size // D
    l0 = (layer_id == 0)
    yv = y_w if l0 else y_v.reshape(n, D)
    vf = v.reshape(n, D) if l0 else v_first.reshape(n, D)
    vbv = a_b if l0 else v_b
    kern = _get_kernel_inside(H, S, l0)
    w, ko, vo, nkk, kka = kern(
        inputs=[y_w.reshape(n, D), y_a.reshape(n, D), yv.reshape(n, D),
                w_b.reshape(D), a_b.reshape(D), vbv.reshape(D),
                k.reshape(n, D), v.reshape(n, D), vf,
                k_k.reshape(D), k_a.reshape(D)],
        grid=(n * H * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n, D)] * 5, output_dtypes=[mx.float32] * 5,
    )
    shp = (B, T, H, S)
    w, ko, vo = w.reshape(shp), ko.reshape(shp), vo.reshape(shp)
    nkk, kka = nkk.reshape(shp), kka.reshape(shp)
    return w, ko, vo, nkk, kka, (vo if l0 else v_first)


# --- ПУСТОЕ ЯДРО: только для замера ---------------------------------------
# Та же геометрия и те же пять выходов, но НИ ОДНОЙ арифметической
# операции -- голое копирование. Меряет цену самого запуска с такой
# раскладкой, отдельно от математики. В decode-путь не подключено.
_cache_e = {}


def prewkv_kernel_empty(y_w, y_a, y_v, k, v, v_first, w_b, a_b, v_b,
                        k_k, k_a, layer_id, B, T, H, S, dtype):
    D = H * S
    n = k.size // D
    key = (H, S)
    if key not in _cache_e:
        header = "\nconstant uint S_C = %d;\n" % S
        body = """
    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint base = tg * S_C;
    for (uint i = lane; i < S_C; i += 32) {
        uint j = base + i;
        wout[j]   = kin[j] + 0.0f * (yw[j] + ya[j] + yv[j]);
        kout[j]   = kin[j];
        vout[j]   = vin[j];
        nkkout[j] = kin[j];
        kkaout[j] = vin[j];
    }
"""
        _cache_e[key] = mx.fast.metal_kernel(
            name="prewkv_empty_h%d_s%d" % (H, S),
            input_names=["kin", "vin", "yw", "ya", "yv"],
            output_names=["wout", "kout", "vout", "nkkout", "kkaout"],
            header=header, source=body)
    w, ko, vo, nkk, kka = _cache_e[key](
        inputs=[k.reshape(n, D), v.reshape(n, D), y_w.reshape(n, D),
                y_a.reshape(n, D),
                (y_w if y_v is None else y_v).reshape(n, D)],
        grid=(n * H * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n, D)] * 5, output_dtypes=[mx.float32] * 5)
    shp = (B, T, H, S)
    return (w.reshape(shp), ko.reshape(shp), vo.reshape(shp),
            nkk.reshape(shp), kka.reshape(shp),
            (vo.reshape(shp) if layer_id == 0 else v_first))


# --- ВАРИАНТ С simd-РЕДУКЦИЕЙ ---------------------------------------------
# Последовательная сумма в prewkv_kernel даёт kk бит-в-бит, но её
# критический путь -- 64 зависимых сложения в каждом лейне. Дерево режет
# путь до ~7 шагов ценой точности: замерено 51.9% бит-в-бит и до 4 ulp,
# то есть relmax порядка 2e-7 при пороге 1e-6 -- проходит с запасом впятеро.
# Подключается, только если ДАСТ ВЫИГРЫШ на замере; иначе смысла терять
# бит-в-бит нет.
_cache_s = {}


def _get_kernel_simd(H, S, layer0):
    key = (H, S, layer0)
    if key in _cache_s:
        return _cache_s[key]
    header = ("\nconstant uint H_C = %d;\nconstant uint S_C = %d;\n"
              "constant bool L0 = %s;\n" % (H, S, "true" if layer0 else "false"))
    body = """
    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint h    = tg % H_C;
    uint base = tg * S_C;
    uint hb   = h * S_C;

    float s2 = 0.0f;
    for (uint i = lane; i < S_C; i += 32) {
        float t  = kin[base + i] * kk_[hb + i];
        float sq = t * t;
        s2 = s2 + sq;
    }
    s2 = simd_sum(s2);
    float r = sqrt(s2 + 1e-12f);

    for (uint i = lane; i < S_C; i += 32) {
        uint j = base + i;
        float zwv = zw[j];
        float sw  = 1.0f / (1.0f + exp(-zwv));
        float ew  = -0.606531f * sw;
        wout[j]   = exp(ew);

        float t   = kin[j] * kk_[hb + i];
        float kkv = t / r;
        nkkout[j] = -kkv;
        float av  = sa[j];
        kkaout[j] = kkv * av;

        float d   = av - 1.0f;
        float p   = d * ka_[hb + i];
        float u   = 1.0f + p;
        kout[j]   = kin[j] * u;

        float v0 = vin[j];
        if (L0) {
            vout[j] = v0;
        } else {
            float dv = vf[j] - v0;
            float tv = dv * sv[j];
            vout[j]  = v0 + tv;
        }
    }
"""
    kern = mx.fast.metal_kernel(
        name="prewkv_simd_h%d_s%d_l%d" % (H, S, int(layer0)),
        input_names=["zw", "sa", "sv", "kin", "vin", "vf", "kk_", "ka_"],
        output_names=["wout", "kout", "vout", "nkkout", "kkaout"],
        header=header, source=body)
    _cache_s[key] = kern
    return kern


def prewkv_kernel_simd(y_w, y_a, y_v, k, v, v_first, w_b, a_b, v_b,
                       k_k, k_a, layer_id, B, T, H, S, dtype):
    D = H * S
    n = k.size // D
    l0 = (layer_id == 0)
    zw = (y_w + w_b).reshape(n, D)
    sa = mx.sigmoid(y_a + a_b).reshape(n, D)
    if l0:
        sv, vf = zw, v.reshape(n, D)
    else:
        sv = mx.sigmoid(y_v + v_b).reshape(n, D)
        vf = v_first.reshape(n, D)
    w, ko, vo, nkk, kka = _get_kernel_simd(H, S, l0)(
        inputs=[zw, sa, sv, k.reshape(n, D), v.reshape(n, D), vf,
                k_k.reshape(D), k_a.reshape(D)],
        grid=(n * H * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n, D)] * 5, output_dtypes=[mx.float32] * 5)
    shp = (B, T, H, S)
    return (w.reshape(shp), ko.reshape(shp), vo.reshape(shp),
            nkk.reshape(shp), kka.reshape(shp), (vo.reshape(shp) if l0 else v_first))
