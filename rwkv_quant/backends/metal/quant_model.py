"""
backends/metal/quant_model.py — сборка RWKV-7 x070 из .rwkvq для MLX-инференса.

Полноранговые Linear-веса (proj: r/k/v/o, cmix: key/value, head) читаются
как QuantLinear напрямую на int8+scale(+SpQR), без разворачивания в bf16.
Всё остальное (LoRA-ветки w/a/v/g, k_k/k_a/r_k, эмбеддинги, LayerNorm/
GroupNorm) держится dense в mx.array — они либо низкоранговые (LoRA, память
копеечная относительно proj/cmix), либо требуют другого паттерна доступа
(emb — gather, не matmul, отдельный кернель вне текущего scope).

Важный нюанс naming="world": «сырые» LoRA-тензоры (w1/w2/a1/a2/...) в
официальных чекпоинтах хранятся в ОБРАТНОЙ ориентации относительно
nn.Linear ([in, out], а не [out, in]) — rwkv7_ref.py транспонирует их
ПОСЛЕ загрузки (см. .T в get(ap+"w1").T и т.д.). writer.py квантует их ДО
этой транспозиции, по сырым ключам state_dict, поэтому per-row scale для
world-LoRA был бы посчитан по input-строкам, а не output-строкам, как для
custom naming. Это не баг здесь — это причина, по которой LoRA-ветки в
этой версии НАРОЧНО оставлены dense: QuantLinear предполагает [out, in],
и для world-LoRA дал бы неверную семантику без отдельного, знающего про
транспозицию кернеля. Квантование LoRA-групп для world-чекпоинтов —
отдельная задача на будущее, не решается тут молча.

WKV-7 рекуррентность НЕ квантуется и берётся как есть из rwkv-metal
(импорт по пути RWKV_METAL_PATH, файлы rwkv-metal не модифицируются).
Используется wkv7_train (не wkv7_infer) — держит произвольный T с
авто-паддингом до CHUNK внутри (16 по умолчанию, было 32 — community:
>16 нестабилен для backward на высокой размерности; см. rwkv-metal#2),
без autograd-накладных расходов при чистом forward (grad просто не
запрашивается). Потоковый inference с персистентным state (wkv7_infer,
T==CHUNK за вызов) — следующий шаг, для scripts/generate.py, не для этого
файла.
"""
import os
import sys

import mlx.core as mx
import torch

from ...formats.reader import _dequantize_one, dequantize_banded  # noqa: F401
from .quant_linear import QuantLinear  # noqa: F401 (v1, референс)
from .quant_linear_v2 import QuantLinearV2
from .quant_linear_gw import GwQuantLinear, GwQuantLinearFused
from .quant_linear_sym import SymQuantLinear, SymQuantLinearFused
from .fused_tail import wkv_tail, can_fuse_tail

# Реализация Linear-кернеля для всей модели. v2 (threadgroup-редукция,
# char4-загрузки) численно эквивалентна v1 (tests/test_quant_linear_v2.py)
# и быстрее на всех shapes 1.5B; v1 остаётся референсом.
_QUANT_LINEAR_IMPL = QuantLinearV2

# Decode-фьюз (стек token-shift лерпов + батч LoRA w/a/v). Веса общие с
# нефьюзнутым путём, вычисления математически эквивалентны (pad нулями
# точен). Переключение в рантайме: qm.FUSE = True/False; компилированный
# step трассирует ветку на момент mx.compile -- после смены флага нужен
# свежий mx.compile.
# ВКЛЮЧЕНО ПО УМОЛЧАНИЮ 16.08 решением владельца проекта. Цена и выигрыш
# измерены и записаны в NEXT_SESSION: фьюз даёт +1.15 мс (6.2%) на
# sym-пресете ценой ~250 МБ резидентно на 1.5B (~425 на 2.9B), гейты
# зелёные, greedy совпадает. Переключается в рантайме, но после смены
# флага нужен свежий mx.compile -- ветка трассируется на момент компиляции.
FUSE = True
LORA_Q8 = False  # int8-лоры в decode-фьюзе: НЕ бит-в-бит, включать после ppl-гейта
# Хвост TMix (group_norm + bonus + gate) одним кернелем вместо ~14 примитивов.
# Действует только внутри фьюзнутого decode-пути и только при FUSE=True;
# редукции там древесно-simd'овые, то есть бит-в-бит не обязан -- гейт
# tests/test_fuse_parity.py.
FUSE_TAIL = True

# ---------------------------------------------------------------------------
# LoRA-ветки под нативный mx.quantized_matmul. ЭТО НОВОЕ КВАНТОВАНИЕ, А НЕ
# РЕПАК: в .rwkvq LoRA лежит в asym gw64, но writer квантует её по СЫРЫМ
# ключам state_dict, до транспозиции, поэтому группы там идут вдоль ВЫХОДНОЙ
# оси матмула, а quantized_matmul требует их вдоль ВХОДНОЙ. Значит меняются и
# значения -- ppl-гейт обязателен (tests/eval_lora_quant.py).
#
#   LORA_Q = None    -- как было, плотный fp16 (умолчание)
#            "sep"   -- четыре down (группы вдоль D, gs=64) и четыре up
#                       (группы вдоль ранга, gs=32) порознь
#            "glue"  -- четыре down ОДНИМ матмулом [R, 2D] по z = [x, xx]
#
# ПРО СКЛЕЙКУ, ЧТОБЫ НЕ ПОВТОРЯТЬ ОШИБКУ ЗАМЕРА. У четырёх down-проекций
# РАЗНЫЕ входы (xw/xa/xv/xg -- x, слитый с token-shift по своему
# коэффициенту), поэтому одного матмула [512, D] с общим входом не
# существует. Точная склейка возможна потому, что x_i = x + xx*c_i:
# матрица [R, 2D] = [A | A*c] по входу z = [x, xx] даёт ровно те же
# значения. Цена -- ВДВОЕ больше строк весов; в fp16 это чистый проигрыш
# (замерено, tests/probe_lora_shapes.py), при шести битах удвоенная
# склейка (1.70 МБ/слой) дешевле нынешнего fp16 порознь (2.10 МБ/слой).
# ВКЛЮЧЕНО ПО УМОЛЧАНИЮ 16.08 решением владельца проекта: +1.32 мс (7.9%)
# на нефьюзнутом пути и +0.60 на фьюзнутом, ppl +0.012% [-0.006; +0.029]
# (не значимо, без размена по языкам), +56 МБ резидентно при живых обеих
# копиях. Восемь бит, а не шесть: по времени они неразличимы (16.65
# против 16.70 мс/ток), а по ppl шесть вшестеро хуже (+0.072%).
LORA_Q = "sep"
LORA_QBITS = 8
LORA_GS_DOWN = 64     # группы вдоль D (2048 -- кратно всегда)
LORA_GS_UP = 32       # группы вдоль ранга (96/64/256 -- 32 делит все три)
# Квантованные ветки -- ТОЛЬКО НА ДЕКОДЕ. При T > 1 они проигрывают
# плотным: замерено сквозным A/B на pp512, 750.4 мс плотными против 789.5
# квантованными на фьюзнутом пути. Причина та же, что у главных матриц:
# при большом N матмул перестаёт быть латентно-связанным, и выигрыш
# кернеля на мелких формах исчезает, а лишние операции распаковки -- нет.
# Побочно это ЛУЧШЕ по качеству: префилл считает LoRA точным деквантом
# файла, квантование живёт только в декоде (его цена измерена отдельно --
# +0.012% ppl, tests/eval_lora_quant.py, гоняется с LORA_Q_DECODE_ONLY=False).
# ГДЕ ИМЕННО ПОРОГ -- НЕ ИЗМЕРЕНО: взята граница T == 1, то есть та же,
# что у фьюза r/k/v. Верификация спекулятивки (N=4) попадёт в плотную
# ветку, и это выбор по умолчанию, а не замер.
LORA_Q_DECODE_ONLY = True

_RWKV_METAL_PATH = os.environ.get("RWKV_METAL_PATH", os.path.expanduser("~/Develop/rwkv-metal"))
if _RWKV_METAL_PATH not in sys.path:
    sys.path.insert(0, _RWKV_METAL_PATH)

from rwkv_metal.kernel.wkv7 import wkv7_train, wkv7_infer, CHUNK  # noqa: E402


def _wkv_stateful(r, w, k, v, a, b, state):
    """Прямой вызов wkv7_infer с произвольным T (>= 1): rwkv-metal с
    параметризованным infer-кернелем (кеш по (H, T)) принимает любой T,
    паддинг/чанкинг больше не нужны. Один путь обслуживает и prefill
    произвольной длины, и single-token decode (T=1) без CHUNKx лишней
    работы, побитово эквивалентно прежнему chunked+padding пути
    (tests/test_wkv_var_model.py: ru60m и 1.5B, max_abs=0.0)."""
    return wkv7_infer(r, w, k, v, a, b, state)


def _dense(qt) -> mx.array:
    """QuantizedTensor -> mx.array. Дороже (полный dequant), для всего,
    что НЕ идёт через QuantLinear.

    2D-матрицы (LoRA A/B и т.п.) храним в fp16: они memory-bound при
    decode, половина трафика; активации остаются fp32, MLX промоутит
    при матмуле. 1D-параметры (LN/GroupNorm, token-shift миксы) — fp32:
    трафик нулевой, точность нормализаций важнее.

    КАСТ ДЕЛАЕТСЯ ДО ВЫХОДА В MLX, И ЭТО НЕ КОСМЕТИКА. Прежний путь
    bf16 -> .float() -> .numpy() -> mx.array -> .astype(fp16) держал
    ЧЕТЫРЕ представления одного тензора и возвращал ЛЕНИВУЮ ноду каста,
    поэтому fp32-копия жила до _materialize(), то есть до конца сборки
    ВСЕЙ модели. На emb 1.5B [65536, 2048] это 1.3 ГБ живых копий ради
    268 МБ результата. Каст в torch даёт ТОТ ЖЕ БИТ (bf16 -> fp32 точен,
    округление ровно одно, fp32 -> fp16) и снимает и копии, и ленивую ноду.
    Деквант при этом идёт полосами строк (reader.dequantize_banded),
    иначе транзиент самого декванта остаётся крупнее результата.
    Гейт равенства обоих путей -- tests/test_dense_load_parity.py."""
    if qt.bits >= 16:
        t = qt.dense
        dt = (torch.float16 if t.ndim == 2 and min(t.shape) >= 32
              else torch.float32)
        return mx.array(t.to(dt).numpy())
    dt = (torch.float16 if len(qt.shape) == 2 and min(qt.shape) >= 32
          else torch.float32)
    return mx.array(dequantize_banded(qt, dt).numpy())


def reset_lora_q(model):
    """Сбросить квантованные LoRA-буферы во всех слоях.

    НУЖНО ВЫЗЫВАТЬ ПОСЛЕ СМЕНЫ LORA_QBITS/LORA_GS_*: буферы строятся ЛЕНИВО
    и кешируются, поэтому смена битности без сброса молча оставляет прежние
    -- ровно та ловушка, на которой споткнулась аблация фьюза (она не видела
    закешированных копий весов). Возвращает число сброшенных слоёв, чтобы
    вызывающий мог убедиться, что сброс вообще что-то нашёл."""
    n = 0
    for b in getattr(model, "blocks", []):
        tm = b.tmix
        if getattr(tm, "_lora_q_built", False):
            n += 1
        tm._lora_q_built = False
        tm._lq_A = tm._lq_B = tm._lq_glue = None
    return n


def drop_lora_dense(model):
    """Освободить fp16-копии LoRA после постройки квантованных.

    ЗАЧЕМ ОТДЕЛЬНОЙ ФУНКЦИЕЙ, А НЕ АВТОМАТОМ ВНУТРИ _build_lora_q. Пока
    обе копии живы, A/B делается подменой ОДНОГО флага в одном процессе
    (закон 27); автоматическое освобождение сделало бы сравнение
    невозможным внутри процесса и загнало бы его в сравнение по
    процессам, где разброс втрое-вчетверо больше (закон 24). Поэтому
    цена включения (обе копии) и цена внедрения (только квантованная)
    -- РАЗНЫЕ числа, и меряются они порознь.

    После вызова fp16-путь LoRA недоступен: LORA_Q обязан остаться
    включённым, иначе _lora упадёт на None вместо тихой подмены."""
    n = 0
    for b in getattr(model, "blocks", []):
        tm = b.tmix
        if getattr(tm, "_lq_A", None) is None:
            raise RuntimeError("сначала _build_lora_q: нечего оставлять "
                               "вместо плотных копий")
        tm._lora_shapes = [(tuple(A.shape), tuple(B.shape))
                           for _, A, B, _ in tm._lora_specs()]
        for attr in ("w_lora_A", "w_lora_B_w", "a_lora_A", "a_lora_B_w",
                     "v_lora_A", "v_lora_B_w", "g_lora_A", "g_lora_B_w"):
            setattr(tm, attr, None)
        tm._dense_lora_dropped = True
        n += 1
    return n


def _mm(x, w):
    """x @ w.T с приведением x к dtype весов (fp16 dense) и результата
    обратно к dtype x. Избегает рантайм-каста весов fp16->fp32 в MLX
    (полный fp32-трафик), который сводил на нет fp16-хранение."""
    return (x.astype(w.dtype) @ w.T).astype(x.dtype)


class _DenseLinear:
    """dense fallback с тем же __call__(x)->y интерфейсом, что и QuantLinear
    — чтобы TMix/CMix не знали, квантован конкретный слой или нет."""
    def __init__(self, w):
        self.w = w  # [out, in]

    def __call__(self, x):
        return _mm(x, self.w)


class MlxAffineQuantLinear:
    """Нативное MLX affine-квантование (mx.nn.QuantizedLinear-совместимый
    формат: uint32-packed codes + fp16 scales/biases per group), НЕ наш
    gw sb6 формат. Добавлено 19.07-10 для сравнения с чужими чекпоинтами
    (MollySophia rwkv7-*-mlx-*bit), квантованными штатным mlx.nn.quantize.
    __call__ через mx.quantized_matmul -- тот же быстрый Metal-путь, что
    использует её собственный рантайм при инференсе (не dense-деквант),
    так что замер скорости честный."""
    def __init__(self, qt):
        self.w = qt.mlx_weight        # uint32 [out, ceil(in*bits/32)]
        self.scales = qt.mlx_scales   # fp16 [out, n_groups]
        self.biases = qt.mlx_biases   # fp16 [out, n_groups]
        self.group_size = qt.mlx_group_size
        self.bits = qt.mlx_bits
        self.in_features = qt.shape[1]
        self.out_features = qt.shape[0]

    def __call__(self, x):
        y = mx.quantized_matmul(x.astype(mx.float16), self.w, scales=self.scales,
                                 biases=self.biases, transpose=True,
                                 group_size=self.group_size, bits=self.bits)
        return y.astype(x.dtype)


def _linear(qt):
    """Linear-подобный тензор [out,in] (proj/cmix/head): QuantLinear если
    реально квантован (bits<16), иначе dense-обёртка с тем же интерфейсом."""
    if qt.bits < 16:
        if getattr(qt, "gw_mode", "") == "sb6":
            return GwQuantLinear(qt)          # формат v2 (gw32 + sb6)
        if getattr(qt, "gw_mode", "") == "sym":
            return SymQuantLinear(qt)         # Q6_K-раскладка, 6 и 8 бит
        if getattr(qt, "gw_mode", "") == "mlx_affine":
            return MlxAffineQuantLinear(qt)   # чужой чекпоинт, нативный MLX affine
        if getattr(qt, "gw_mode", "") == "asym":
            # gw-asym (LoRA-класс) как linear не встречается: LoRA идут
            # dense-путём (_dense -> _dequantize_one). Если попали сюда --
            # деквант в fp16-dense, чтобы не падать.
            return _DenseLinear(mx.array(
                dequantize_banded(qt, torch.float16).numpy()))
        return _QUANT_LINEAR_IMPL(qt)
    return _DenseLinear(mx.array(qt.dense.to(torch.float16).numpy()))


def l2_norm(x):
    return x / mx.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-12)


def _group_norm(x, H, weight, bias, eps=64e-5):
    # x: [N, D], normalize per group of size D//H, как F.group_norm(num_groups=H)
    N, D = x.shape
    S = D // H
    xg = x.reshape(N, H, S)
    mean = xg.mean(axis=-1, keepdims=True)
    var = ((xg - mean) ** 2).mean(axis=-1, keepdims=True)
    xg = (xg - mean) / mx.sqrt(var + eps)
    xg = xg.reshape(N, D) * weight + bias
    return xg


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps) * weight + bias


def _token_shift(x, x_prev=None):
    pad = mx.zeros_like(x[:, :1]) if x_prev is None else x_prev
    return mx.concatenate([pad, x[:, :-1]], axis=1)


def _token_shift_stateful(x, prev):
    """Как _token_shift, но дополнительно возвращает новый prev (последний
    x текущего вызова) -- для переноса через границу single-token вызовов
    в streaming decode. prev=None на первом вызове == поведение _token_shift
    без state (нулевой pad)."""
    B, T, D = x.shape
    p = mx.zeros((B, 1, D)) if prev is None else prev
    shifted = mx.concatenate([p, x[:, :-1]], axis=1)
    new_prev = x[:, -1:]
    return shifted, new_prev


class QuantTMix:
    # на классе, а не только в __init__: подклассы вроде MollyTMix
    # (tests/eval_molly_real.py) собирают поля сами и зовут _build_fused
    # напрямую, минуя наш конструктор
    _fused_built = False

    def __init__(self, tensors, layer_prefix, naming, layer_id, n_head, head_size):
        self.H, self.S = n_head, head_size
        self.layer_id = layer_id

        def g(suffix):
            return tensors[layer_prefix + suffix]

        if naming == "custom":
            tp = "tmix."
            self.x_r, self.x_w, self.x_k = _dense(g(tp+"x_r")), _dense(g(tp+"x_w")), _dense(g(tp+"x_k"))
            self.x_v, self.x_a, self.x_g = _dense(g(tp+"x_v")), _dense(g(tp+"x_a")), _dense(g(tp+"x_g"))
            self.w_lora_A = _dense(g(tp+"w_lora_A.weight"))
            self.w_lora_B_w = _dense(g(tp+"w_lora_B.weight")); self.w_lora_B_b = _dense(g(tp+"w_lora_B.bias"))
            self.a_lora_A = _dense(g(tp+"a_lora_A.weight"))
            self.a_lora_B_w = _dense(g(tp+"a_lora_B.weight")); self.a_lora_B_b = _dense(g(tp+"a_lora_B.bias"))
            self.v_lora_A = self.v_lora_B_w = self.v_lora_B_b = None
            if layer_id > 0:
                self.v_lora_A = _dense(g(tp+"v_lora_A.weight"))
                self.v_lora_B_w = _dense(g(tp+"v_lora_B.weight")); self.v_lora_B_b = _dense(g(tp+"v_lora_B.bias"))
            self.g_lora_A = _dense(g(tp+"g_lora_A.weight"))
            self.g_lora_B_w = _dense(g(tp+"g_lora_B.weight"))
            self.k_k = _dense(g(tp+"k_k")).reshape(self.H, self.S)
            self.k_a = _dense(g(tp+"k_a")).reshape(self.H, self.S)
            self.r_k = _dense(g(tp+"r_k")).reshape(self.H, self.S)
            self.r_proj = _linear(g(tp+"r_proj.weight")); self.k_proj = _linear(g(tp+"k_proj.weight"))
            self.v_proj = _linear(g(tp+"v_proj.weight")); self.o_proj = _linear(g(tp+"o_proj.weight"))
            self.ln_x_w, self.ln_x_b = _dense(g(tp+"ln_x.weight")), _dense(g(tp+"ln_x.bias"))
            self._build_fused()
        else:
            ap = "att."
            self.x_r, self.x_w, self.x_k = _dense(g(ap+"x_r")), _dense(g(ap+"x_w")), _dense(g(ap+"x_k"))
            self.x_v, self.x_a, self.x_g = _dense(g(ap+"x_v")), _dense(g(ap+"x_a")), _dense(g(ap+"x_g"))
            self.w_lora_A = _dense(g(ap+"w1")).T
            self.w_lora_B_w = _dense(g(ap+"w2")).T; self.w_lora_B_b = _dense(g(ap+"w0")).reshape(-1)
            self.a_lora_A = _dense(g(ap+"a1")).T
            self.a_lora_B_w = _dense(g(ap+"a2")).T; self.a_lora_B_b = _dense(g(ap+"a0")).reshape(-1)
            self.v_lora_A = self.v_lora_B_w = self.v_lora_B_b = None
            if layer_id > 0:
                self.v_lora_A = _dense(g(ap+"v1")).T
                self.v_lora_B_w = _dense(g(ap+"v2")).T; self.v_lora_B_b = _dense(g(ap+"v0")).reshape(-1)
            self.g_lora_A = _dense(g(ap+"g1")).T
            self.g_lora_B_w = _dense(g(ap+"g2")).T
            self.k_k = _dense(g(ap+"k_k")).reshape(self.H, self.S)
            self.k_a = _dense(g(ap+"k_a")).reshape(self.H, self.S)
            self.r_k = _dense(g(ap+"r_k")).reshape(self.H, self.S)
            self.r_proj = _linear(g(ap+"receptance.weight")); self.k_proj = _linear(g(ap+"key.weight"))
            self.v_proj = _linear(g(ap+"value.weight")); self.o_proj = _linear(g(ap+"output.weight"))
            self.ln_x_w, self.ln_x_b = _dense(g(ap+"ln_x.weight")), _dense(g(ap+"ln_x.bias"))
        self._fused_built = False

    def __call__(self, x, v_first):
        B, T, D = x.shape
        H, S = self.H, self.S

        xx = _token_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.r_proj(xr).reshape(B, T, H, S)
        k = self.k_proj(xk).reshape(B, T, H, S)
        v = self.v_proj(xv).reshape(B, T, H, S)

        y_w, y_a, y_v, g = self._lora(xw, xa, xv, xg, x, xx)

        a = mx.sigmoid(y_a + self.a_lora_B_b)
        a = a.reshape(B, T, H, S)

        w = y_w + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(y_v + self.v_lora_B_b).reshape(B, T, H, S)
            v = v + (v_first - v) * vv

        out = wkv7_train(r, w, k, v, -kk, kk * a)

        out2d = out.reshape(B * T, D)
        out2d = _group_norm(out2d, H, self.ln_x_w, self.ln_x_b)
        out = out2d.reshape(B, T, H, S)
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out = (out + bonus).reshape(B, T, D)

        return self.o_proj(out * g), v_first

    def _lora_specs(self):
        """(имя, A [r, D], B [D, r], коэффициент лерпа) по веткам в порядке
        w, a, v, g. Слой 0 без v -- ветка просто отсутствует в списке, а не
        зануляется: паддинг тут ничего не экономит, а лишние строки читаются
        как настоящие."""
        out = [("w", self.w_lora_A, self.w_lora_B_w, self.x_w),
               ("a", self.a_lora_A, self.a_lora_B_w, self.x_a)]
        if self.v_lora_A is not None:
            out.append(("v", self.v_lora_A, self.v_lora_B_w, self.x_v))
        out.append(("g", self.g_lora_A, self.g_lora_B_w, self.x_g))
        return out

    def _build_lora_q(self):
        """Квантованные буферы LoRA. Строятся ЛЕНИВО и живут ПОВЕРХ fp16-копий:
        нефьюзнутый путь, префилл и гейты продолжают читать плотные веса, то
        есть A/B делается подменой флага в ОДНОМ процессе (закон 27).

        Не строим и молча остаёмся на fp16, если формы не кратны группе
        (toy-модели): mx.quantize требует кратности, а тихо сменить группу
        значило бы померить не ту раскладку."""
        if getattr(self, "_lora_q_built", False):
            return
        self._lora_q_built = True
        self._lq_A = self._lq_B = self._lq_glue = None
        specs = self._lora_specs()
        D = int(self.x_r.shape[-1])
        bits = LORA_QBITS
        if D % LORA_GS_DOWN or D % LORA_GS_UP:
            return
        if any(int(A.shape[0]) % LORA_GS_UP for _, A, _, _ in specs):
            return
        ev = []
        self._lq_names = [nm for nm, _, _, _ in specs]
        self._lq_A = [mx.quantize(A, group_size=LORA_GS_DOWN, bits=bits)
                      for _, A, _, _ in specs]
        self._lq_B = [mx.quantize(B, group_size=LORA_GS_UP, bits=bits)
                      for _, _, B, _ in specs]
        # склейка: [R, 2D] = [A | A*c], группы не пересекают границу D
        # (gs=64 делит D), поэтому левая половина квантуется ровно так же,
        # как в варианте "sep".
        rows = []
        for _, A, _, c in specs:
            c32 = c.reshape(1, -1).astype(mx.float32)
            rows.append(mx.concatenate(
                [A.astype(mx.float32), A.astype(mx.float32) * c32], axis=1))
        glue = mx.contiguous(mx.concatenate(rows, axis=0))          # [R, 2D]
        self._lq_glue = mx.quantize(glue, group_size=LORA_GS_DOWN, bits=bits)
        self._lq_slices = []
        off = 0
        for _, A, _, _ in specs:
            r = int(A.shape[0])
            self._lq_slices.append((off, off + r))
            off += r
        del glue, rows
        for tr in (self._lq_A + self._lq_B + [self._lq_glue]):
            ev.extend(tr)
        mx.eval(ev)

    def _lora(self, xw, xa, xv, xg, x, xx):
        """Все LoRA-ветки слоя: возвращает (y_w, y_a, y_v, y_g) -- выходы
        ПОСЛЕ up-проекции, ДО прибавления bias и внешних нелинейностей.
        y_v = None на нулевом слое.

        Одна реализация на все три пути (prefill __call__, forward_stateful,
        фьюзнутый): иначе правка, внесённая в один, в остальные не переезжает
        сама собой -- закон 23."""
        mode = LORA_Q
        if (mode and LORA_Q_DECODE_ONLY
                and not getattr(self, "_dense_lora_dropped", False)
                and x.size // x.shape[-1] > 1):
            mode = None                      # префилл идёт плотным путём
        if mode and not getattr(self, "_lora_q_built", False):
            self._build_lora_q()
        if mode and self._lq_A is None:
            mode = None                      # формы не кратны -- честный fp16
        if mode:
            names = self._lq_names
            specs = [(nm, None, None, None) for nm in names]
        else:
            if getattr(self, "_dense_lora_dropped", False):
                raise RuntimeError(
                    "плотные копии LoRA освобождены (drop_lora_dense), "
                    "а LORA_Q выключен -- fp16-пути больше нет")
            specs = self._lora_specs()
            names = [s[0] for s in specs]

        xin = {"w": xw, "a": xa, "v": xv, "g": xg}
        if mode == "glue":
            z = mx.concatenate([x, xx], axis=-1).astype(mx.float16)
            wq, sc, bi = self._lq_glue
            hcat = mx.quantized_matmul(z, wq, scales=sc, biases=bi,
                                       transpose=True, group_size=LORA_GS_DOWN,
                                       bits=LORA_QBITS).astype(x.dtype)
            hs = [hcat[..., a:b] for a, b in self._lq_slices]
        elif mode == "sep":
            hs = []
            for i, nm in enumerate(names):
                wq, sc, bi = self._lq_A[i]
                hs.append(mx.quantized_matmul(
                    xin[nm].astype(mx.float16), wq, scales=sc, biases=bi,
                    transpose=True, group_size=LORA_GS_DOWN,
                    bits=LORA_QBITS).astype(x.dtype))
        else:
            hs = [_mm(xin[nm], A) for nm, A, _, _ in specs]

        ys = []
        for i, nm in enumerate(names):
            B = None if mode else specs[i][2]
            h = hs[i]
            if nm == "w":
                h = mx.tanh(h)
            elif nm == "g":
                h = mx.sigmoid(h)
            if mode:
                wq, sc, bi = self._lq_B[i]
                ys.append(mx.quantized_matmul(
                    h.astype(mx.float16), wq, scales=sc, biases=bi,
                    transpose=True, group_size=LORA_GS_UP,
                    bits=LORA_QBITS).astype(h.dtype))
            else:
                ys.append(_mm(h, B))
        out = dict(zip(names, ys))
        return out["w"], out["a"], out.get("v"), out["g"]

    def _build_fused(self):
        """Буферы decode-фьюза: [6,1,1,D]-стек лерп-коэффициентов и
        батченые LoRA-матрицы (w,a,v): pad v-ранга (64->96) нулями --
        нулевые строки/столбцы дают точный ноль, эквивалентность полная.
        g (ранг 256) не батчится -- остаётся парой отдельных матмулов.
        Слой 0 без v-ветки: v-слот нулевой, его выход не используется.

        СТРОИТСЯ ЛЕНИВО, ПО ПЕРВОМУ ВХОДУ В ФЬЮЗНУТЫЙ ПУТЬ. Прежде он
        строился в конструкторе БЕЗУСЛОВНО, а FUSE по умолчанию False --
        то есть дубль r/k/v (250 МБ на 1.5B, 425 на 2.9B) и стеки лор
        лежали в памяти мёртвым грузом у всех, кто фьюз не включал, то
        есть у всех замеров по умолчанию. Именно этим, а не раскладкой,
        объяснялось «пресет с sym на 227 МБ ЛЕГЧЕ»: при proj в sym фьюз
        просто не строился (GwQuantLinearFused требует GwQuantLinear).
        Ленивость сохраняет семантику флага целиком -- FUSE и LORA_Q8
        остаются переключаемыми в рантайме, просто платит за них тот,
        кто включил."""
        if self._fused_built:
            return
        self._fused_built = True
        D = self.x_r.shape[-1]
        self.xcoef = mx.stack([self.x_r, self.x_w, self.x_k,
                               self.x_v, self.x_a, self.x_g])  # [6,1,1,D]
        self._wav_idx = mx.array([1, 4, 3])          # (xw, xa, xv) из xs
        self._tanh_mask = mx.array([True, False, False]).reshape(3, 1, 1)
        self._wav_built = False
        self.wav_At = self.wav_Bt = None
        self._wav_At_q = self._wav_Bt_q = None
        self._g_A_q = self._g_B_q = None
        self._build_rkv_fused()
        mx.eval([self.xcoef])

    def _build_wav(self):
        """Батченые стеки LoRA (w,a,v) фьюзнутого пути. ОТДЕЛЬНО ОТ
        _build_fused и лениво, потому что при LORA_Q они не читаются
        вовсе: там ветки идут порознь квантованными, и паддинг ранга
        v (64 -> 96), ради которого батч и заводился, стал бы чистым
        убытком. На 1.5B это 57.8 МБ, которые иначе лежали бы мёртвым
        грузом ровно так же, как до 15.08 лежал дубль r/k/v."""
        if self._wav_built:
            return
        self._wav_built = True
        D = self.x_r.shape[-1]
        rs = [self.w_lora_A.shape[0], self.a_lora_A.shape[0]]
        if self.v_lora_A is not None:
            rs.append(self.v_lora_A.shape[0])
        rmax = max(rs)

        def padA(A):
            if A is None:
                return mx.zeros((rmax, D), dtype=self.w_lora_A.dtype)
            if A.shape[0] == rmax:
                return A
            return mx.concatenate(
                [A, mx.zeros((rmax - A.shape[0], D), dtype=A.dtype)], axis=0)

        def padBt(Bw):  # Bw [D, r] -> Bt [rmax, D]
            if Bw is None:
                return mx.zeros((rmax, D), dtype=self.w_lora_B_w.dtype)
            Bt = Bw.T
            if Bt.shape[0] == rmax:
                return Bt
            return mx.concatenate(
                [Bt, mx.zeros((rmax - Bt.shape[0], D), dtype=Bt.dtype)], axis=0)

        self.wav_At = mx.stack([padA(self.w_lora_A), padA(self.a_lora_A),
                                padA(self.v_lora_A)]).transpose(0, 2, 1)  # [3,D,rmax]
        self.wav_Bt = mx.stack([padBt(self.w_lora_B_w), padBt(self.a_lora_B_w),
                                padBt(self.v_lora_B_w)])                  # [3,rmax,D]
        self._wav_idx = mx.array([1, 4, 3])          # (xw, xa, xv) из xs
        # int8-копии для LORA_Q8 (трафик лор пополам; значения --
        # mx.quantize(gs=64, bits=8) поверх fp16-стеков). Итог 19.07:
        # decode x1.004 (лора-блок латентно-, а не трафик-bound) --
        # отрицательный результат, флаг выключен. Строим только когда
        # размерности кратны 64 (toy-модели -- нет).
        self._wav_At_q = self._wav_Bt_q = None
        self._g_A_q = self._g_B_q = None
        if (D % 64 == 0 and rmax % 64 == 0
                and self.g_lora_A.shape[-1] % 64 == 0
                and self.g_lora_B_w.shape[-1] % 64 == 0):
            self._wav_At_q = mx.quantize(
                mx.contiguous(self.wav_At.transpose(0, 2, 1)),
                group_size=64, bits=8)                               # [3,rmax,D]
            self._wav_Bt_q = mx.quantize(self.wav_Bt, group_size=64, bits=8)
            self._g_A_q = mx.quantize(self.g_lora_A, group_size=64, bits=8)
            self._g_B_q = mx.quantize(self.g_lora_B_w, group_size=64, bits=8)
        ev = [self.wav_At, self.wav_Bt]
        ev += [a for pair in (self._wav_At_q, self._wav_Bt_q,
                              self._g_A_q, self._g_B_q) if pair is not None
               for a in pair]
        mx.eval(ev)

    def _build_rkv_fused(self):
        # r/k/v одним launch'ем: конкатенация квантованных строк трёх
        # GwQuantLinear (формат нетронут, математика строки бит-в-бит).
        # Цена: копия буферов (~8.7MB/слой) поверх оригиналов -- оригиналы
        # нужны GEMM-префиллу и нефьюзнутому пути.
        self._rkv_fused = None
        self._rkv_idx = mx.array([0, 2, 3])          # (xr, xk, xv) из xs
        lins = [self.r_proj, self.k_proj, self.v_proj]
        if (all(isinstance(l, GwQuantLinear) for l in lins)
                and len({(l.in_features, l.out_features, l.has_qh)
                         for l in lins}) == 1):
            self._rkv_fused = GwQuantLinearFused(lins)
        elif (all(isinstance(l, SymQuantLinear) for l in lins)
                and all(lins[0].can_fuse_with(l) for l in lins)):
            # раскладка sym: без этой ветки целевой пресет (proj в sym)
            # фьюза не получал ВООБЩЕ -- он молча не строился
            self._rkv_fused = SymQuantLinearFused(lins)

    def _forward_stateful_fused(self, x, v_first, state):
        """forward_stateful с decode-фьюзом: 6 лерпов -> 1 broadcast-оп;
        LoRA (w,a,v) -> 2 batched-матмула вместо 6. Математика идентична
        нефьюзнутому пути (см. _build_fused)."""
        if not self._fused_built:
            self._build_fused()
        wkv_state, shift_state = state
        B, T, D = x.shape
        H, S = self.H, self.S

        shifted, new_shift_state = _token_shift_stateful(x, shift_state)
        xx = shifted - x
        xs = x[None] + xx[None] * self.xcoef          # [6,B,T,D]
        xg = xs[5]

        if self._rkv_fused is not None and B * T == 1:
            rkv = self._rkv_fused(mx.take(xs, self._rkv_idx, axis=0).reshape(3, D))
            r = rkv[0].reshape(B, T, H, S)
            k = rkv[1].reshape(B, T, H, S)
            v = rkv[2].reshape(B, T, H, S)
        else:
            xr, xk, xv = xs[0], xs[2], xs[3]
            r = self.r_proj(xr).reshape(B, T, H, S)
            k = self.k_proj(xk).reshape(B, T, H, S)
            v = self.v_proj(xv).reshape(B, T, H, S)

        if LORA_Q:
            # квантованные ветки: батченые стеки wav не строятся и не
            # читаются -- у них паддинг ранга v (64 -> 96), то есть лишние
            # байты, ради которых батч и заводился, когда байты были fp16
            y_w, y_a, y_v, g = self._lora(xs[1], xs[4], xs[3], xs[5], x, xx)
            y_w = y_w.reshape(B, T, D)
            y_a = y_a.reshape(B, T, D)
            y_v = None if y_v is None else y_v.reshape(B, T, D)
        else:
            if LORA_Q8 and self._g_A_q is not None:
                gq = mx.quantized_matmul(
                    xg.astype(mx.float16), *self._g_A_q, transpose=True,
                    group_size=64, bits=8)
                g = mx.quantized_matmul(
                    mx.sigmoid(gq), *self._g_B_q, transpose=True,
                    group_size=64, bits=8).astype(x.dtype)
            else:
                g = (_mm(mx.sigmoid(_mm(xg, self.g_lora_A)), self.g_lora_B_w))

            self._build_wav()
            z = mx.take(xs, self._wav_idx, axis=0).reshape(3, B * T, D)
            if LORA_Q8 and self._wav_At_q is not None:
                h = mx.quantized_matmul(
                    z.astype(mx.float16), *self._wav_At_q, transpose=True,
                    group_size=64, bits=8).astype(x.dtype)
                h = mx.where(self._tanh_mask, mx.tanh(h), h)
                y = mx.quantized_matmul(
                    h.astype(mx.float16), *self._wav_Bt_q, transpose=False,
                    group_size=64, bits=8).astype(x.dtype)  # [3,BT,D]
            else:
                h = (z.astype(self.wav_At.dtype) @ self.wav_At).astype(x.dtype)
                h = mx.where(self._tanh_mask, mx.tanh(h), h)
                y = (h.astype(self.wav_Bt.dtype) @ self.wav_Bt).astype(x.dtype)  # [3,BT,D]
            y_w, y_a = y[0].reshape(B, T, D), y[1].reshape(B, T, D)
            y_v = y[2].reshape(B, T, D)

        w = y_w + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)
        a = mx.sigmoid(y_a + self.a_lora_B_b).reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(y_v + self.v_lora_B_b).reshape(B, T, H, S)
            v = v + (v_first - v) * vv

        out, new_wkv_state = _wkv_stateful(r, w, k, v, -kk, kk * a, wkv_state)

        if FUSE_TAIL and can_fuse_tail(H, S):
            # group_norm + bonus + gate одним запуском вместо ~14
            out = wkv_tail(out, r, k, v, self.r_k, self.ln_x_w, self.ln_x_b,
                           g, H, S).reshape(B, T, D)
            return self.o_proj(out), v_first, (new_wkv_state, new_shift_state)

        out2d = out.reshape(B * T, D)
        out2d = _group_norm(out2d, H, self.ln_x_w, self.ln_x_b)
        out = out2d.reshape(B, T, H, S)
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out = (out + bonus).reshape(B, T, D)

        return self.o_proj(out * g), v_first, (new_wkv_state, new_shift_state)

    def forward_stateful(self, x, v_first, state):
        """То же самое, что __call__, но WKV-рекуррентность идёт через
        _wkv_stateful (chunked wkv7_infer + переносимый state) вместо
        wkv7_train. Используется для streaming prefill/decode.

        state = (wkv_state, shift_state): wkv_state -- [B,H,S,S] численный
        state рекуррентности; shift_state -- последний x предыдущего
        вызова (None на первом вызове), нужен для token-shift на границе
        отдельных single-token decode-вызовов."""
        if FUSE:
            return self._forward_stateful_fused(x, v_first, state)
        wkv_state, shift_state = state
        B, T, D = x.shape
        H, S = self.H, self.S

        shifted, new_shift_state = _token_shift_stateful(x, shift_state)
        xx = shifted - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.r_proj(xr).reshape(B, T, H, S)
        k = self.k_proj(xk).reshape(B, T, H, S)
        v = self.v_proj(xv).reshape(B, T, H, S)

        y_w, y_a, y_v, g = self._lora(xw, xa, xv, xg, x, xx)

        a = mx.sigmoid(y_a + self.a_lora_B_b)
        a = a.reshape(B, T, H, S)

        w = y_w + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(y_v + self.v_lora_B_b).reshape(B, T, H, S)
            v = v + (v_first - v) * vv

        out, new_wkv_state = _wkv_stateful(r, w, k, v, -kk, kk * a, wkv_state)

        out2d = out.reshape(B * T, D)
        out2d = _group_norm(out2d, H, self.ln_x_w, self.ln_x_b)
        out = out2d.reshape(B, T, H, S)
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out = (out + bonus).reshape(B, T, D)

        return self.o_proj(out * g), v_first, (new_wkv_state, new_shift_state)


class QuantCMix:
    def __init__(self, tensors, layer_prefix, naming):
        def g(suffix):
            return tensors[layer_prefix + suffix]
        cp = "cmix." if naming == "custom" else "ffn."
        self.x_k = _dense(g(cp+"x_k"))
        self.key = _linear(g(cp+"key.weight"))
        self.value = _linear(g(cp+"value.weight"))

    def __call__(self, x):
        xx = _token_shift(x) - x
        xk = x + xx * self.x_k
        h = self.key(xk)
        h = mx.maximum(h, 0.0) ** 2
        return self.value(h)

    def forward_stateful(self, x, shift_state):
        shifted, new_shift_state = _token_shift_stateful(x, shift_state)
        xx = shifted - x
        xk = x + xx * self.x_k
        h = self.key(xk)
        h = mx.maximum(h, 0.0) ** 2
        return self.value(h), new_shift_state


class QuantBlock:
    def __init__(self, tensors, layer_prefix, naming, layer_id, n_head, head_size):
        def g(suffix):
            return tensors[layer_prefix + suffix]
        self.ln1_w, self.ln1_b = _dense(g("ln1.weight")), _dense(g("ln1.bias"))
        self.ln2_w, self.ln2_b = _dense(g("ln2.weight")), _dense(g("ln2.bias"))
        self.tmix = QuantTMix(tensors, layer_prefix, naming, layer_id, n_head, head_size)
        self.cmix = QuantCMix(tensors, layer_prefix, naming)

    def __call__(self, x, v_first):
        h, v_first = self.tmix(_layer_norm(x, self.ln1_w, self.ln1_b), v_first)
        x = x + h
        x = x + self.cmix(_layer_norm(x, self.ln2_w, self.ln2_b))
        return x, v_first

    def step(self, x, v_first, state):
        # state = (wkv_state, tmix_shift, cmix_shift)
        wkv_state, tmix_shift, cmix_shift = state
        h, v_first, (new_wkv_state, new_tmix_shift) = self.tmix.forward_stateful(
            _layer_norm(x, self.ln1_w, self.ln1_b), v_first, (wkv_state, tmix_shift))
        x = x + h
        cmix_out, new_cmix_shift = self.cmix.forward_stateful(
            _layer_norm(x, self.ln2_w, self.ln2_b), cmix_shift)
        x = x + cmix_out
        return x, v_first, (new_wkv_state, new_tmix_shift, new_cmix_shift)


class QuantRWKV7:
    """RWKV-7 x070 forward (prefill, T произвольный) на .rwkvq через MLX.
    Строится напрямую из QuantizedCheckpoint (formats.reader.load_raw)."""

    def __init__(self, ckpt):
        # ckpt: rwkv_quant.formats.schema.QuantizedCheckpoint
        self.naming = ckpt.naming
        self.n_layer = ckpt.n_layer
        self.n_embd = ckpt.n_embd
        self.head_size = ckpt.head_size
        self.n_head = ckpt.n_embd // ckpt.head_size
        self.vocab_size = ckpt.vocab_size
        tensors = ckpt.tensors

        self.emb_weight = _dense(tensors["emb.weight"])   # gather, всегда dense
        self.head = _linear(tensors["head.weight"])

        if self.naming == "custom":
            self.ln0_w, self.ln0_b = _dense(tensors["ln0.weight"]), _dense(tensors["ln0.bias"])
        else:
            self.ln0_w, self.ln0_b = _dense(tensors["blocks.0.ln0.weight"]), _dense(tensors["blocks.0.ln0.bias"])
        self.ln_out_w, self.ln_out_b = _dense(tensors["ln_out.weight"]), _dense(tensors["ln_out.bias"])

        self.blocks = [
            QuantBlock(tensors, f"blocks.{i}.", self.naming, i, self.n_head, self.head_size)
            for i in range(self.n_layer)
        ]
        self._materialize()

    def _materialize(self):
        """Принудительный eval всех параметров. КРИТИЧНО для mx.compile:
        _dense() строит ленивые astype-ноды (fp32->fp16); если mx.compile
        трассирует шаг ДО их материализации, касты захватываются в граф и
        пересчитываются на КАЖДЫЙ вызов (fp32-трафик всех dense-весов:
        16 vs 26 мс/ток на COMPRESSION, бистабильность зависела от того,
        успел ли eager-вызов материализовать веса до первого model.step)."""
        arrs = []
        def collect(obj, depth=0):
            if depth > 3:
                return
            for v in vars(obj).values():
                if isinstance(v, mx.array):
                    arrs.append(v)
                elif isinstance(v, (list, tuple)):
                    for it in v:
                        if isinstance(it, mx.array):
                            arrs.append(it)
                        elif hasattr(it, "__dict__"):
                            collect(it, depth+1)
                elif hasattr(v, "__dict__"):
                    collect(v, depth+1)
        collect(self)
        mx.eval(arrs)

    def __call__(self, idx: mx.array) -> mx.array:
        x = self.emb_weight[idx]
        x = _layer_norm(x, self.ln0_w, self.ln0_b)
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = _layer_norm(x, self.ln_out_w, self.ln_out_b)
        return self.head(x)

    def init_state(self, batch_size: int = 1):
        """Нулевой per-layer state под streaming: список [n_layer] из
        (wkv_state, tmix_shift, cmix_shift). wkv_state -- mx.array [B,H,S,S]
        (то, что ждёт wkv7_infer как h_in); tmix_shift/cmix_shift -- None
        (= нулевой pad на первом вызове, как в не-streaming __call__)."""
        H, S = self.n_head, self.head_size
        return [(mx.zeros((batch_size, H, S, S)), None, None) for _ in range(self.n_layer)]

    @property
    def step(self):
        """СКОМПИЛИРОВАННЫЙ вход в модель -- и для декода, И ДЛЯ ПРЕФИЛЛА.

        `mx.compile` фьюзит elementwise-цепочки и кеширует граф по формам,
        поэтому T=1 и T=512 живут отдельными кешами внутри ОДНОГО
        объекта: звать это `step` исторически, но для префилла оно
        годится ровно так же, и звать сырой `forward_stateful` там
        незачем.

        Замерено 15.08 (`tests/bench_compile_ab.py`, 1.5B, sym-пресет,
        чередование, разброс 0.3-0.5%):

            префилл pp512  344.3 -> 532.5 ток/с   (+35.3%)
            декод          43.8 -> 52.8 ток/с     (+17.0%)

        ЦЕНА -- ТРАССИРОВКА НА КАЖДУЮ НОВУЮ ФОРМУ: 406 мс на первую
        длину префилла, 110 мс на каждую следующую, 33 мс на T=1. Она
        окупается НЕМЕДЛЕННО, а не в среднем: один префилл на 512 токенов
        экономит 525 мс (1487 -> 962), то есть даже первый вызов на
        незнакомой длине выходит на 415 мс быстрее сырого пути. Для
        декода окупаемость -- девять токенов.
        Чинить трассировку кусками фиксированной формы ПРОБОВАЛИ и это
        хуже: куски по 256 стоят 26% скорости, по 128 -- 86%
        (`tests/bench_prefill_chunk_ab.py`). `shapeless=True` на этом
        графе не работает вовсе -- `Slice cannot infer output shapes`.

        Численно: rel ~3e-4 (порядок fp16-шума), greedy-траектория
        64 токенов идентична eager (tests/verify_compile.py). Для
        отладки/сверок использовать сырой forward_stateful."""
        if not hasattr(self, "_step_compiled"):
            self._step_compiled = mx.compile(self.forward_stateful)
        return self._step_compiled

    def forward_stateful(self, idx: mx.array, states, last_only: bool = False,
                         tail_only: int = 0):
        """idx: [B, T] -- T=1 для single-token decode, T>1 для prefill
        произвольной длины (внутри чанкуется по 32 автоматически).
        states: список per-layer state из init_state() или предыдущего
        вызова. Возвращает (logits, new_states).

        last_only=True: head считается только для последней позиции
        (logits [B, 1, V]) -- для prefill в генерации, где нужен лишь
        следующий токен, это убирает (T-1)/T работы head'а (65536x2048 на
        1.5B). Дефолт False сохраняет полные логиты (ppl, тесты)."""
        x = self.emb_weight[idx]
        x = _layer_norm(x, self.ln0_w, self.ln0_b)
        v_first = None
        new_states = []
        for block, state in zip(self.blocks, states):
            x, v_first, new_state = block.step(x, v_first, state)
            new_states.append(new_state)
        x = _layer_norm(x, self.ln_out_w, self.ln_out_b)
        if last_only and x.shape[1] > 1:
            x = x[:, -1:]
        elif tail_only and x.shape[1] > tail_only:
            # спекулятивная верификация: логиты нужны только на tail_only
            # хвостовых позициях (k драфт + 1 бонус); pending-префикс
            # двигает state, но head по нему не считается (19.07-15, п.1)
            x = x[:, -tail_only:]
        return self.head(x), new_states
