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
авто-паддингом до CHUNK=32 внутри, без autograd-накладных расходов при
чистом forward (grad просто не запрашивается). Потоковый inference с
персистентным state (wkv7_infer, T==32 за вызов) — следующий шаг, для
scripts/generate.py, не для этого файла.
"""
import os
import sys

import mlx.core as mx

from ...formats.reader import _dequantize_one
from .quant_linear import QuantLinear

_RWKV_METAL_PATH = os.environ.get("RWKV_METAL_PATH", os.path.expanduser("~/Develop/rwkv-metal"))
if _RWKV_METAL_PATH not in sys.path:
    sys.path.insert(0, _RWKV_METAL_PATH)

from rwkv_metal.kernel.wkv7 import wkv7_train, wkv7_infer, CHUNK  # noqa: E402


def _wkv_stateful(r, w, k, v, a, b, state):
    """Прогоняет произвольный T через wkv7_infer чанками по CHUNK=32,
    неся state между вызовами. Хвостовой неполный чанк паддится no-op
    шагами (w=1, v=0, a=0 -> state*1 + 0 + 0 = state, доказано по формуле
    рекуррентности в kernel/wkv7.py:279). Один и тот же путь обслуживает
    и prefill произвольной длины, и single-token decode (T=1) -- разница
    только в T на входе, код один."""
    B, T, H, S = r.shape
    outs = []
    pos = 0
    while pos < T:
        n = min(CHUNK, T - pos)
        r_c = r[:, pos:pos+n]; w_c = w[:, pos:pos+n]; k_c = k[:, pos:pos+n]
        v_c = v[:, pos:pos+n]; a_c = a[:, pos:pos+n]; b_c = b[:, pos:pos+n]
        if n < CHUNK:
            pad = CHUNK - n
            zeros = mx.zeros((B, pad, H, S))
            ones = mx.ones((B, pad, H, S))
            r_c = mx.concatenate([r_c, zeros], axis=1)
            w_c = mx.concatenate([w_c, ones], axis=1)
            k_c = mx.concatenate([k_c, zeros], axis=1)
            v_c = mx.concatenate([v_c, zeros], axis=1)
            a_c = mx.concatenate([a_c, zeros], axis=1)
            b_c = mx.concatenate([b_c, zeros], axis=1)
        out_c, state = wkv7_infer(r_c, w_c, k_c, v_c, a_c, b_c, state)
        outs.append(out_c[:, :n])
        pos += n
    return mx.concatenate(outs, axis=1), state


def _dense(qt) -> mx.array:
    """QuantizedTensor -> mx.array. Дороже (полный dequant), для всего,
    что НЕ идёт через QuantLinear."""
    t = _dequantize_one(qt) if qt.bits < 16 else qt.dense
    return mx.array(t.float().numpy())


class _DenseLinear:
    """dense fallback с тем же __call__(x)->y интерфейсом, что и QuantLinear
    — чтобы TMix/CMix не знали, квантован конкретный слой или нет."""
    def __init__(self, w):
        self.w = w  # [out, in]

    def __call__(self, x):
        return x @ self.w.T


def _linear(qt):
    """Linear-подобный тензор [out,in] (proj/cmix/head): QuantLinear если
    реально квантован (bits<16), иначе dense-обёртка с тем же интерфейсом."""
    if qt.bits < 16:
        return QuantLinear(qt)
    return _DenseLinear(mx.array(qt.dense.float().numpy()))


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

        g = (mx.sigmoid(xg @ self.g_lora_A.T) @ self.g_lora_B_w.T)

        a = mx.sigmoid((xa @ self.a_lora_A.T) @ self.a_lora_B_w.T + self.a_lora_B_b)
        a = a.reshape(B, T, H, S)

        w = (mx.tanh(xw @ self.w_lora_A.T)) @ self.w_lora_B_w.T + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid((xv @ self.v_lora_A.T) @ self.v_lora_B_w.T + self.v_lora_B_b).reshape(B, T, H, S)
            v = v + (v_first - v) * vv

        out = wkv7_train(r, w, k, v, -kk, kk * a)

        out2d = out.reshape(B * T, D)
        out2d = _group_norm(out2d, H, self.ln_x_w, self.ln_x_b)
        out = out2d.reshape(B, T, H, S)
        bonus = (r * k * self.r_k).sum(axis=-1, keepdims=True) * v
        out = (out + bonus).reshape(B, T, D)

        return self.o_proj(out * g), v_first

    def forward_stateful(self, x, v_first, state):
        """То же самое, что __call__, но WKV-рекуррентность идёт через
        _wkv_stateful (chunked wkv7_infer + переносимый state) вместо
        wkv7_train. Используется для streaming prefill/decode.

        state = (wkv_state, shift_state): wkv_state -- [B,H,S,S] численный
        state рекуррентности; shift_state -- последний x предыдущего
        вызова (None на первом вызове), нужен для token-shift на границе
        отдельных single-token decode-вызовов."""
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

        g = (mx.sigmoid(xg @ self.g_lora_A.T) @ self.g_lora_B_w.T)

        a = mx.sigmoid((xa @ self.a_lora_A.T) @ self.a_lora_B_w.T + self.a_lora_B_b)
        a = a.reshape(B, T, H, S)

        w = (mx.tanh(xw @ self.w_lora_A.T)) @ self.w_lora_B_w.T + self.w_lora_B_b
        w = mx.exp(-0.606531 * mx.sigmoid(w.astype(mx.float32))).astype(x.dtype)
        w = w.reshape(B, T, H, S)

        kk = l2_norm(k * self.k_k)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        if self.layer_id == 0:
            v_first = v
        else:
            vv = mx.sigmoid((xv @ self.v_lora_A.T) @ self.v_lora_B_w.T + self.v_lora_B_b).reshape(B, T, H, S)
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

    def forward_stateful(self, idx: mx.array, states):
        """idx: [B, T] -- T=1 для single-token decode, T>1 для prefill
        произвольной длины (внутри чанкуется по 32 автоматически).
        states: список per-layer state из init_state() или предыдущего
        вызова. Возвращает (logits, new_states)."""
        x = self.emb_weight[idx]
        x = _layer_norm(x, self.ln0_w, self.ln0_b)
        v_first = None
        new_states = []
        for block, state in zip(self.blocks, states):
            x, v_first, new_state = block.step(x, v_first, state)
            new_states.append(new_state)
        x = _layer_norm(x, self.ln_out_w, self.ln_out_b)
        return self.head(x), new_states
