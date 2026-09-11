"""Проба: воспроизводимы ли примитивы MLX в Metal ПОБИТОВО.

ЗАЧЕМ. Пороги приёмки ядра пред-WKV на fp16-выходах (relmax 1e-6 на w,
kk, kk*a и бит-в-бит на k, v) лежат НИЖЕ решётки fp16: один ulp на
элементе масштаба единицы даёт относительные 4.9e-4. Значит порог
проверяет не точность, а совпадение бит, и вопрос до написания ядра
ровно один -- достижимо ли такое совпадение вообще. Проба меряет это
по трём примитивам ОТДЕЛЬНО, на РЕАЛЬНЫХ входах модели:

    sigmoid в fp16     -- решает судьбу порогов на k и v;
    sigmoid+exp в fp32 -- решает судьбу порога на w;
    l2_norm по 64      -- решает судьбу порогов на kk и kk*a.

Мерится доля побитово совпавших элементов и максимум расхождения В ULP,
а не в относительных: относительные на fp16 обманывают, потому что
сравнивают с решёткой типа.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
import rwkv_quant.backends.metal.quant_model as qm

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Develop/WKV-kvant/compression_0p1b.rwkvq")
CORPUS = os.path.expanduser("~/Develop/WKV-kvant/eval_corpus_multiling.pt")

model = qm.QuantRWKV7(load_raw(MODEL))
TOK = torch.load(CORPUS)["tokens"].numpy().astype(np.int32)

CAP = []
_o = qm.QuantTMix._prewkv


def _c(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype):
    CAP.append((y_w + self.w_lora_B_b, y_a + self.a_lora_B_b,
                k * self.k_k, self.H, self.S, k, self.k_a, v,
                v_first, None if y_v is None else y_v + self.v_lora_B_b))
    return _o(self, y_w, y_a, y_v, k, v, v_first, B, T, dtype)


qm.QuantTMix._prewkv = _c
st = model.init_state(1)
lg, st = model.forward_stateful(mx.array(TOK[0:1, :64]), st, last_only=True)
mx.eval(lg)
CAP.clear()
tk = mx.argmax(lg[:, -1], axis=-1)
lg, st = model.forward_stateful(tk.reshape(1, 1), st)
mx.eval(lg)
qm.QuantTMix._prewkv = _o

H, S = CAP[0][3], CAP[0][4]
ZW = mx.concatenate([c[0].reshape(-1) for c in CAP])
ZA = mx.concatenate([c[1].reshape(-1) for c in CAP])
KI = mx.concatenate([c[2].reshape(-1, S) for c in CAP])
mx.eval(ZW, ZA, KI)
print("захвачено слоёв %d, H=%d S=%d; ZW %r %s, ZA %s, KI %r %s" % (
    len(CAP), H, S, ZW.shape, ZW.dtype, ZA.dtype, KI.shape,
    KI.dtype), flush=True)


def ulps(got, ref):
    # ulp МЕРЯЕТСЯ В ТИПЕ ЭТАЛОНА. Первый заход мерил в fp16 при fp32-
    # пути и завышал совпадение: приведение к fp16 само стирает
    # расхождения младших разрядов (закон 39 -- линейка грубее
    # измеряемого).
    dt = np.float32 if ref.dtype == mx.float32 else np.float16
    it = np.int32 if dt is np.float32 else np.int16
    sign = np.int64(-2147483648) if dt is np.float32 else np.int64(-32768)
    a = np.array(got.astype(ref.dtype)).astype(dt)
    b = np.array(ref).astype(dt)
    ia = a.view(it).astype(np.int64)
    ib = b.view(it).astype(np.int64)
    ia = np.where(ia < 0, sign - ia, ia)
    ib = np.where(ib < 0, sign - ib, ib)
    return np.abs(ia - ib)


def report(tag, got, ref):
    u = ulps(got, ref)
    eq = float((u == 0).mean())
    print("  %-26s побитово %8.5f%%   max %d ulp   ne %d" % (
        tag, 100.0 * eq, int(u.max()), int((u != 0).sum())), flush=True)
    return eq


def unary(name, src, x, out_dtype):
    n = x.size
    hdr = "constant uint N_C = %d;\n" % n
    body = ("    uint i = thread_position_in_grid.x;\n"
            "    if (i >= N_C) return;\n" + src)
    k = mx.fast.metal_kernel(name=name, input_names=["inp"],
                             output_names=["out"], header=hdr, source=body)
    g = ((n + 255) // 256) * 256
    return k(inputs=[x], grid=(g, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[x.shape], output_dtypes=[out_dtype])[0]


# --- 1. sigmoid (решает k и v) ------------------------------------------
print("1. sigmoid, эталон mx.sigmoid, %d элементов, тип %s" % (
      ZA.size, ZA.dtype), flush=True)
ref1 = mx.sigmoid(ZA)
mx.eval(ref1)
SIG = {
    "1/(1+exp(-x))":
        "    float x = inp[i];\n    out[i] = 1.0f / (1.0f + exp(-x));\n",
    "устойчивая по знаку":
        "    float x = inp[i];\n"
        "    float y = 1.0f / (1.0f + exp(-fabs(x)));\n"
        "    out[i] = (x < 0.0f) ? (1.0f - y) : y;\n",
    "precise::exp":
        "    float x = inp[i];\n"
        "    out[i] = 1.0f / (1.0f + metal::precise::exp(-x));\n",
    "fast::exp":
        "    float x = inp[i];\n"
        "    out[i] = 1.0f / (1.0f + metal::fast::exp(-x));\n",
    "precise::divide":
        "    float x = inp[i];\n"
        "    out[i] = metal::precise::divide(1.0f, 1.0f + exp(-x));\n",
    "через exp(x)":
        "    float x = inp[i];\n    float e = exp(x);\n"
        "    out[i] = e / (1.0f + e);\n",
}
for j, (tag, src) in enumerate(SIG.items()):
    try:
        got = unary("sig_%d" % j, src, ZA, ZA.dtype)
        mx.eval(got)
        report(tag, got, ref1)
    except Exception as e:
        print("  %-26s ОШИБКА %s" % (tag, str(e)[:90]), flush=True)


# --- 2. w = exp(-0.606531*sigmoid(x)) (решает w) ------------------------
print("2. w, эталон mx.exp(-0.606531*mx.sigmoid(fp32)), %d элементов" %
      ZW.size, flush=True)
ref2 = mx.exp(-0.606531 * mx.sigmoid(ZW.astype(mx.float32)))
mx.eval(ref2)
WCAND = {
    "exp / exp":
        "    float x = inp[i];\n"
        "    float s = 1.0f / (1.0f + exp(-x));\n"
        "    out[i] = exp(-0.606531f * s);\n",
    "precise::exp оба":
        "    float x = inp[i];\n"
        "    float s = 1.0f / (1.0f + metal::precise::exp(-x));\n"
        "    out[i] = metal::precise::exp(-0.606531f * s);\n",
    "fast::exp оба":
        "    float x = inp[i];\n"
        "    float s = 1.0f / (1.0f + metal::fast::exp(-x));\n"
        "    out[i] = metal::fast::exp(-0.606531f * s);\n",
}
for j, (tag, src) in enumerate(WCAND.items()):
    try:
        got = unary("wc_%d" % j, src, ZW, ref2.dtype)
        mx.eval(got)
        report(tag, got, ref2)
    except Exception as e:
        print("  %-26s ОШИБКА %s" % (tag, str(e)[:90]), flush=True)


# --- 3. l2_norm по строке из S (решает kk и kk*a) -----------------------
print("3. l2_norm по %d, эталон qm.l2_norm, %d строк, тип %s" % (
      S, KI.shape[0], KI.dtype), flush=True)
ref3 = qm.l2_norm(KI)
mx.eval(ref3)


def rowk(name, src, x):
    n, s = x.shape
    hdr = "constant uint S_C = %d;\n" % s
    body = ("    uint tg = threadgroup_position_in_grid.x;\n"
            "    uint lane = thread_position_in_threadgroup.x;\n"
            "    uint base = tg * S_C;\n" + src)
    k = mx.fast.metal_kernel(name=name, input_names=["inp"],
                             output_names=["out"], header=hdr, source=body)
    return k(inputs=[x], grid=(n * 32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


ACC_SIMD = ("    float s2 = 0.0f;\n"
            "    for (uint i = lane; i < S_C; i += 32) {\n"
            "        float t = inp[base + i]; s2 += t * t; }\n"
            "    s2 = simd_sum(s2);\n")
ACC_SEQ = ("    float s2 = 0.0f;\n"
           "    for (uint i = 0; i < S_C; ++i) {\n"
           "        float t = inp[base + i]; s2 += t * t; }\n")
DIV = ("    float r = sqrt(s2 + 1e-12f);\n"
       "    for (uint i = lane; i < S_C; i += 32)\n"
       "        out[base + i] = inp[base + i] / r;\n")
RSQ = ("    float r = rsqrt(s2 + 1e-12f);\n"
       "    for (uint i = lane; i < S_C; i += 32)\n"
       "        out[base + i] = inp[base + i] * r;\n")
PDIV = ("    float r = metal::precise::sqrt(s2 + 1e-12f);\n"
        "    for (uint i = lane; i < S_C; i += 32)\n"
        "        out[base + i] = metal::precise::divide(inp[base + i], r);\n")
L2 = {
    "simd f32, деление": ACC_SIMD + DIV,
    "simd f32, rsqrt": ACC_SIMD + RSQ,
    "simd f32, precise": ACC_SIMD + PDIV,
    "последовательно, деление": ACC_SEQ + DIV,
    "последовательно, rsqrt": ACC_SEQ + RSQ,
    "последовательно, precise": ACC_SEQ + PDIV,
}
for j, (tag, src) in enumerate(L2.items()):
    try:
        got = rowk("l2_%d" % j, src, KI)
        mx.eval(got)
        report(tag, got, ref3)
    except Exception as e:
        print("  %-26s ОШИБКА %s" % (tag, str(e)[:90]), flush=True)


# --- 4. поэлементные цепочки fp32: ловит ли компилятор contraction в fma
# ЗАЧЕМ. Если sigmoid отдавать в ядро ГОТОВЫМ (посчитанным MLX), то k и v
# становятся чистой цепочкой умножений и сложений, и бит-в-бит на них
# достижим -- НО только если Metal не свернёт a*b+c в fma. Свёртка даёт
# другой результат (одно округление вместо двух), и это ровно тот 1 ulp,
# который порог бит-в-бит не прощает. Здесь это и меряется.
L0 = [c for c in CAP if c[9] is not None][0]
KR, KA = L0[5].reshape(-1), L0[6].reshape(-1)
A1 = mx.sigmoid(L0[1]).reshape(-1)
VR, VF, ZV = L0[7].reshape(-1), L0[8].reshape(-1), L0[9].reshape(-1)
VV = mx.sigmoid(ZV)
mx.eval(KR, KA, A1, VR, VF, VV)
print("4. цепочки fp32 на готовых sigmoid, %d элементов" % KR.size,
      flush=True)


def tri(tag, src, xs, ref, _n=[0]):
    n = xs[0].size
    hdr = "constant uint N_C = %d;\n" % n
    body = ("    uint i = thread_position_in_grid.x;\n"
            "    if (i >= N_C) return;\n" + src)
    _n[0] += 1
    k = mx.fast.metal_kernel(name="chain_%d" % _n[0], input_names=["p", "q", "r"],
                             output_names=["out"], header=hdr, source=body)
    g = ((n + 255) // 256) * 256
    got = k(inputs=list(xs), grid=(g, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[xs[0].shape], output_dtypes=[ref.dtype])[0]
    mx.eval(got)
    return report(tag, got, ref)


kref = KR * (1.0 + (A1 - 1.0) * KA)
vref = VR + (VF - VR) * VV
mx.eval(kref, vref)
tri("k: как записано",
    "    out[i] = p[i] * (1.0f + (q[i] - 1.0f) * r[i]);\n",
    (KR, A1, KA), kref)
tri("k: по шагам",
    "    float t = (q[i] - 1.0f) * r[i];\n"
    "    float u = 1.0f + t;\n    out[i] = p[i] * u;\n",
    (KR, A1, KA), kref)
tri("k: явная fma",
    "    out[i] = p[i] * fma(q[i] - 1.0f, r[i], 1.0f);\n",
    (KR, A1, KA), kref)
tri("v: как записано",
    "    out[i] = p[i] + (q[i] - p[i]) * r[i];\n",
    (VR, VF, VV), vref)
tri("v: по шагам",
    "    float d = q[i] - p[i];\n"
    "    float t = d * r[i];\n    out[i] = p[i] + t;\n",
    (VR, VF, VV), vref)
tri("v: явная fma",
    "    out[i] = fma(q[i] - p[i], r[i], p[i]);\n",
    (VR, VF, VV), vref)
print("ГОТОВО", flush=True)
