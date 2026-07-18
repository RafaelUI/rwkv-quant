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

from ...formats.reader import _dequantize_one
from .quant_linear import QuantLinear  # noqa: F401 (v1, референс)
from .quant_linear_v2 import QuantLinearV2
from .quant_linear_gw import GwQuantLinear, GwQuantLinearFused

# Реализация Linear-кернеля для всей модели. v2 (threadgroup-редукция,
# char4-загрузки) численно эквивалентна v1 (tests/test_quant_linear_v2.py)
# и быстрее на всех shapes 1.5B; v1 остаётся референсом.
_QUANT_LINEAR_IMPL = QuantLinearV2

# Decode-фьюз (стек token-shift лерпов + батч LoRA w/a/v). Веса общие с
# нефьюзнутым путём, вычисления математически эквивалентны (pad нулями
# точен). Переключение в рантайме: qm.FUSE = True/False; компилированный
# step трассирует ветку на момент mx.compile -- после смены флага нужен
# свежий mx.compile.
FUSE = False

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
    трафик нулевой, точность нормализаций важнее."""
    t = _dequantize_one(qt) if qt.bits < 16 else qt.dense
    arr = mx.array(t.float().numpy())
    if arr.ndim == 2 and min(arr.shape) >= 32:
        return arr.astype(mx.float16)
    return arr


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


def _linear(qt):
    """Linear-подобный тензор [out,in] (proj/cmix/head): QuantLinear если
    реально квантован (bits<16), иначе dense-обёртка с тем же интерфейсом."""
    if qt.bits < 16:
        if getattr(qt, "gw_mode", "") == "sb6":
            return GwQuantLinear(qt)          # формат v2 (gw32 + sb6)
        if getattr(qt, "gw_mode", "") == "asym":
            # gw-asym (LoRA-класс) как linear не встречается: LoRA идут
            # dense-путём (_dense -> _dequantize_one). Если попали сюда --
            # деквант в fp16-dense, чтобы не падать.
            from ...formats.reader import _dequantize_one
            return _DenseLinear(mx.array(_dequantize_one(qt).float().numpy()).astype(mx.float16))
        return _QUANT_LINEAR_IMPL(qt)
    return _DenseLinear(mx.array(qt.dense.float().numpy()).astype(mx.float16))


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
            self._build_fused()

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

        g = (_mm(mx.sigmoid(_mm(xg, self.g_lora_A)), self.g_lora_B_w))

        a = mx.sigmoid(_mm(_mm(xa, self.a_lora_A), self.a_lora_B_w) + self.a_lora_B_b)
        a = a.reshape(B, T, H, S)

        w = _mm(mx.tanh(_mm(xw, self.w_lora_A)), self.w_lora_B_w) + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(_mm(_mm(xv, self.v_lora_A), self.v_lora_B_w) + self.v_lora_B_b).reshape(B, T, H, S)
            v = v + (v_first - v) * vv

        out = wkv7_train(r, w, k, v, -kk, kk * a)

        out2d = out.reshape(B * T, D)
        out2d = _group_norm(out2d, H, self.ln_x_w, self.ln_x_b)
        out = out2d.reshape(B, T, H, S)
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out = (out + bonus).reshape(B, T, D)

        return self.o_proj(out * g), v_first

    def _build_fused(self):
        """Буферы decode-фьюза: [6,1,1,D]-стек лерп-коэффициентов и
        батченые LoRA-матрицы (w,a,v): pad v-ранга (64->96) нулями --
        нулевые строки/столбцы дают точный ноль, эквивалентность полная.
        g (ранг 256) не батчится -- остаётся парой отдельных матмулов.
        Слой 0 без v-ветки: v-слот нулевой, его выход не используется."""
        D = self.x_r.shape[-1]
        self.xcoef = mx.stack([self.x_r, self.x_w, self.x_k,
                               self.x_v, self.x_a, self.x_g])  # [6,1,1,D]
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
        self._tanh_mask = mx.array([True, False, False]).reshape(3, 1, 1)

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

    def _forward_stateful_fused(self, x, v_first, state):
        """forward_stateful с decode-фьюзом: 6 лерпов -> 1 broadcast-оп;
        LoRA (w,a,v) -> 2 batched-матмула вместо 6. Математика идентична
        нефьюзнутому пути (см. _build_fused)."""
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

        g = (_mm(mx.sigmoid(_mm(xg, self.g_lora_A)), self.g_lora_B_w))

        z = mx.take(xs, self._wav_idx, axis=0).reshape(3, B * T, D)
        h = (z.astype(self.wav_At.dtype) @ self.wav_At).astype(x.dtype)
        h = mx.where(self._tanh_mask, mx.tanh(h), h)
        y = (h.astype(self.wav_Bt.dtype) @ self.wav_Bt).astype(x.dtype)  # [3,BT,D]

        w = y[0].reshape(B, T, D) + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)
        a = mx.sigmoid(y[1].reshape(B, T, D) + self.a_lora_B_b).reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(y[2].reshape(B, T, D) + self.v_lora_B_b).reshape(B, T, H, S)
            v = v + (v_first - v) * vv

        out, new_wkv_state = _wkv_stateful(r, w, k, v, -kk, kk * a, wkv_state)

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

        g = (_mm(mx.sigmoid(_mm(xg, self.g_lora_A)), self.g_lora_B_w))

        a = mx.sigmoid(_mm(_mm(xa, self.a_lora_A), self.a_lora_B_w) + self.a_lora_B_b)
        a = a.reshape(B, T, H, S)

        w = _mm(mx.tanh(_mm(xw, self.w_lora_A)), self.w_lora_B_w) + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid(_mm(_mm(xv, self.v_lora_A), self.v_lora_B_w) + self.v_lora_B_b).reshape(B, T, H, S)
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
        """Скомпилированный forward_stateful для decode-циклов: mx.compile
        фьюзит elementwise-цепочки и кеширует граф по shapes (префилл и
        T=1 живут отдельными кешами). На 1.5B REDUCTION: 22.3 -> 19.8 мс/ток
        (+13%). Численно: rel ~3e-4 (порядок fp16-шума), greedy-траектория
        64 токенов идентична eager (tests/verify_compile.py). Для
        отладки/сверок использовать сырой forward_stateful."""
        if not hasattr(self, "_step_compiled"):
            self._step_compiled = mx.compile(self.forward_stateful)
        return self._step_compiled

    def forward_stateful(self, idx: mx.array, states, last_only: bool = False):
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
        return self.head(x), new_states
