"""
Референсная реализация forward pass RWKV-7 (PyTorch), используется ТОЛЬКО
для калибровки/валидации квантования -- НЕ прод-инференс путь.

Понимает обе схемы именования чекпоинтов (см. naming.py) и приводит их к
единому внутреннему представлению (TMix/CMix), поэтому forward-математика
(и вся quantization-логика через calibration.q()) написана один раз и
работает для любого чекпоинта RWKV-7 независимо от того, как он обучен.

Медленная реализация WKV-рекуррентности (O(T) python-цикл) сделана
намеренно: приоритет корректности над скоростью, т.к. от результатов
калибровки напрямую зависят решения о квантовании, идущие в backends/.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..calibration import QuantConfig, q
from .naming import detect_naming


class TMix:
    __slots__ = (
        "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
        "w_lora_A", "w_lora_B_w", "w_lora_B_b",
        "a_lora_A", "a_lora_B_w", "a_lora_B_b",
        "v_lora_A", "v_lora_B_w", "v_lora_B_b",
        "g_lora_A", "g_lora_B_w",
        "k_k", "k_a", "r_k",
        "r_proj", "k_proj", "v_proj", "o_proj",
        "ln_x_w", "ln_x_b",
        "n_head", "head_size",
    )


class CMix:
    __slots__ = ("x_k", "key", "value")


class RWKV7Ref(nn.Module):
    def __init__(self, ckpt_path: str, device="mps", dtype=torch.bfloat16,
                 n_layer=None, n_embd=None, head_size=64, vocab_size=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        naming = detect_naming(ckpt_path, None)
        if naming == "world":
            sd = torch.load(ckpt_path, map_location="cpu")
        else:
            from safetensors.torch import load_file
            sd = load_file(f"{ckpt_path}/model.safetensors")
            with open(f"{ckpt_path}/config.json") as f:
                cfg = json.load(f)
            n_layer = cfg["n_layer"]; n_embd = cfg["n_embd"]
            head_size = cfg["head_size"]; vocab_size = cfg["vocab_size"]

        if naming == "world":
            import re
            layers = set()
            for k in sd.keys():
                m = re.match(r"blocks\.(\d+)\.", k)
                if m:
                    layers.add(int(m.group(1)))
            n_layer = max(layers) + 1
            n_embd = sd["emb.weight"].shape[1]
            vocab_size = sd["emb.weight"].shape[0]

        self.n_layer, self.n_embd = n_layer, n_embd
        self.head_size = head_size
        self.n_head = n_embd // head_size
        self.vocab_size = vocab_size
        self.naming = naming
        print(f"[RWKV7Ref] naming={naming} n_layer={n_layer} n_embd={n_embd} "
              f"n_head={self.n_head} head_size={head_size} vocab={vocab_size}")

        def get(name):
            return sd[name].to(device=device, dtype=dtype)

        if naming == "custom":
            self.emb_weight = get("emb.weight")
            self.head_weight = get("head.weight")
            self.ln0_w, self.ln0_b = get("ln0.weight"), get("ln0.bias")
            self.ln_out_w, self.ln_out_b = get("ln_out.weight"), get("ln_out.bias")
        else:
            self.emb_weight = get("emb.weight")
            self.head_weight = get("head.weight")
            self.ln0_w, self.ln0_b = get("blocks.0.ln0.weight"), get("blocks.0.ln0.bias")
            self.ln_out_w, self.ln_out_b = get("ln_out.weight"), get("ln_out.bias")

        self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b = [], [], [], []
        self.tmix, self.cmix = [], []

        for i in range(self.n_layer):
            p = f"blocks.{i}."
            self.ln1_w.append(get(p + "ln1.weight")); self.ln1_b.append(get(p + "ln1.bias"))
            self.ln2_w.append(get(p + "ln2.weight")); self.ln2_b.append(get(p + "ln2.bias"))

            t = TMix()
            if naming == "custom":
                tp = p + "tmix."
                t.x_r, t.x_w, t.x_k = get(tp + "x_r"), get(tp + "x_w"), get(tp + "x_k")
                t.x_v, t.x_a, t.x_g = get(tp + "x_v"), get(tp + "x_a"), get(tp + "x_g")
                t.w_lora_A = get(tp + "w_lora_A.weight")
                t.w_lora_B_w = get(tp + "w_lora_B.weight"); t.w_lora_B_b = get(tp + "w_lora_B.bias")
                t.a_lora_A = get(tp + "a_lora_A.weight")
                t.a_lora_B_w = get(tp + "a_lora_B.weight"); t.a_lora_B_b = get(tp + "a_lora_B.bias")
                if (tp + "v_lora_A.weight") in sd:
                    t.v_lora_A = get(tp + "v_lora_A.weight")
                    t.v_lora_B_w = get(tp + "v_lora_B.weight"); t.v_lora_B_b = get(tp + "v_lora_B.bias")
                else:
                    t.v_lora_A = t.v_lora_B_w = t.v_lora_B_b = None
                t.g_lora_A = get(tp + "g_lora_A.weight")
                t.g_lora_B_w = get(tp + "g_lora_B.weight")
                t.k_k = get(tp + "k_k"); t.k_a = get(tp + "k_a"); t.r_k = get(tp + "r_k")
                t.r_proj = get(tp + "r_proj.weight"); t.k_proj = get(tp + "k_proj.weight")
                t.v_proj = get(tp + "v_proj.weight"); t.o_proj = get(tp + "o_proj.weight")
                t.ln_x_w = get(tp + "ln_x.weight"); t.ln_x_b = get(tp + "ln_x.bias")
            else:
                ap = p + "att."
                t.x_r, t.x_w, t.x_k = get(ap + "x_r"), get(ap + "x_w"), get(ap + "x_k")
                t.x_v, t.x_a, t.x_g = get(ap + "x_v"), get(ap + "x_a"), get(ap + "x_g")
                # world stores LoRA A/B as raw (in,out)/(out',in') matmul weights + separate "0" bias
                t.w_lora_A = get(ap + "w1").T.contiguous()
                t.w_lora_B_w = get(ap + "w2").T.contiguous()
                t.w_lora_B_b = get(ap + "w0").reshape(-1).contiguous()
                t.a_lora_A = get(ap + "a1").T.contiguous()
                t.a_lora_B_w = get(ap + "a2").T.contiguous()
                t.a_lora_B_b = get(ap + "a0").reshape(-1).contiguous()
                if i == 0:
                    t.v_lora_A = t.v_lora_B_w = t.v_lora_B_b = None  # unused at layer 0, per reference
                else:
                    t.v_lora_A = get(ap + "v1").T.contiguous()
                    t.v_lora_B_w = get(ap + "v2").T.contiguous()
                    t.v_lora_B_b = get(ap + "v0").reshape(-1).contiguous()
                t.g_lora_A = get(ap + "g1").T.contiguous()
                t.g_lora_B_w = get(ap + "g2").T.contiguous()
                t.k_k = get(ap + "k_k"); t.k_a = get(ap + "k_a"); t.r_k = get(ap + "r_k")
                t.r_proj = get(ap + "receptance.weight"); t.k_proj = get(ap + "key.weight")
                t.v_proj = get(ap + "value.weight"); t.o_proj = get(ap + "output.weight")
                t.ln_x_w = get(ap + "ln_x.weight"); t.ln_x_b = get(ap + "ln_x.bias")

            t.n_head, t.head_size = self.n_head, self.head_size
            self.tmix.append(t)

            c = CMix()
            if naming == "custom":
                cp = p + "cmix."
                c.x_k = get(cp + "x_k"); c.key = get(cp + "key.weight"); c.value = get(cp + "value.weight")
            else:
                fp = p + "ffn."
                c.x_k = get(fp + "x_k"); c.key = get(fp + "key.weight"); c.value = get(fp + "value.weight")
            self.cmix.append(c)

    @staticmethod
    def _time_shift(x):
        return F.pad(x, (0, 0, 1, -1))

    def _tmix_forward(self, x, v_first, t: TMix, layer_id: int, cfg: QuantConfig):
        B, T, C = x.shape
        H, N = t.n_head, t.head_size
        xx = self._time_shift(x) - x

        xr = x + xx * t.x_r
        xw = x + xx * t.x_w
        xk = x + xx * t.x_k
        xv = x + xx * t.x_v
        xa = x + xx * t.x_a
        xg = x + xx * t.x_g

        r = xr @ q(t.r_proj, "proj", cfg).T
        w_ = -F.softplus(-(F.linear(torch.tanh(xw @ q(t.w_lora_A, "w_lora", cfg).T),
                                     q(t.w_lora_B_w, "w_lora", cfg), t.w_lora_B_b))) - 0.5
        k = xk @ q(t.k_proj, "proj", cfg).T
        v = xv @ q(t.v_proj, "proj", cfg).T

        if layer_id == 0:
            v_first = v
        else:
            resid_gate = torch.sigmoid(F.linear(xv @ q(t.v_lora_A, "v_lora", cfg).T,
                                                 q(t.v_lora_B_w, "v_lora", cfg), t.v_lora_B_b))
            v = v + (v_first - v) * resid_gate

        a = torch.sigmoid(F.linear(xa @ q(t.a_lora_A, "a_lora", cfg).T,
                                    q(t.a_lora_B_w, "a_lora", cfg), t.a_lora_B_b))
        g = torch.sigmoid(xg @ q(t.g_lora_A, "g_lora", cfg).T) @ q(t.g_lora_B_w, "g_lora", cfg).T

        k_k = q(t.k_k, "small", cfg).reshape(1, 1, C)
        k_a = q(t.k_a, "small", cfg).reshape(1, 1, C)
        r_k = q(t.r_k, "small", cfg).reshape(H, N)

        kk = k * k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * k_a)

        out = self._wkv7(r, w_, k, v, -kk, kk * a, H, N)
        out = F.group_norm(out.view(B * T, C).float(), H, t.ln_x_w.float(), t.ln_x_b.float(),
                            eps=64e-5).view(B, T, C).to(x.dtype)

        bonus = ((r.view(B, T, H, N) * k.view(B, T, H, N) * r_k).sum(dim=-1, keepdim=True)
                 * v.view(B, T, H, N)).view(B, T, C)
        out = out + bonus
        out = (out * g) @ q(t.o_proj, "proj", cfg).T
        return out, v_first

    @staticmethod
    def _wkv7(r, w, k, v, a, b, H, N):
        B, T, C = r.shape
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
        for tt in range(T):
            kk = k[:, tt, :].view(B, H, 1, N)
            rr = r[:, tt, :].view(B, H, N, 1)
            vv = v[:, tt, :].view(B, H, N, 1)
            aa = a[:, tt, :].view(B, H, N, 1)
            bb = b[:, tt, :].view(B, H, 1, N)
            state = state * w[:, tt, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, tt, :] = (state @ rr).view(B, H, N)
        return out.view(B, T, C)

    def _cmix_forward(self, x, c: CMix, cfg: QuantConfig):
        xx = self._time_shift(x) - x
        k = x + xx * c.x_k
        k = torch.relu(k @ q(c.key, "cmix", cfg).T) ** 2
        return k @ q(c.value, "cmix", cfg).T

    def forward(self, idx: torch.Tensor, cfg: QuantConfig = None):
        if cfg is None:
            cfg = QuantConfig()
        x = F.embedding(idx, q(self.emb_weight, "emb_head", cfg))
        x = F.layer_norm(x.float(), (self.n_embd,), self.ln0_w.float(), self.ln0_b.float()).to(x.dtype)

        v_first = torch.empty_like(x)
        for i in range(self.n_layer):
            xn = F.layer_norm(x.float(), (self.n_embd,), self.ln1_w[i].float(), self.ln1_b[i].float()).to(x.dtype)
            att, v_first = self._tmix_forward(xn, v_first, self.tmix[i], i, cfg)
            x = x + att
            xn2 = F.layer_norm(x.float(), (self.n_embd,), self.ln2_w[i].float(), self.ln2_b[i].float()).to(x.dtype)
            x = x + self._cmix_forward(xn2, self.cmix[i], cfg)

        x = F.layer_norm(x.float(), (self.n_embd,), self.ln_out_w.float(), self.ln_out_b.float()).to(x.dtype)
        logits = x @ q(self.head_weight, "emb_head", cfg).T
        return logits
