"""Восстановление имён параметров из тренировочного чекпоинта.

В снимке лежит `master_params` -- ПЛОСКИЙ список тензоров в порядке
named_parameters(), без имён. Имена восстанавливаются генерацией того же
порядка и СВЕРКОЙ ФОРМ: если хоть одна не совпала, файл не пишется вовсе.
Это и делает восстановление проверяемым, а не правдоподобным.

Размерности БЕРУТСЯ ИЗ САМОГО ЧЕКПОИНТА, а не зашиты: emb идёт первым,
из него C и словарь; ранги LoRA -- из форм первого блока. Зашитые
константы описывали ровно один чекпоинт, и второй такой же
архитектуры пришлось бы копировать файлом (закон 23: копии расходятся).

    python tests_local/build_world_ckpt_v2.py <src.pt> <out.pth>
"""
import sys

import torch

SRC = sys.argv[1] if len(sys.argv) > 1 else '/Users/s/Develop/ckpt_step135000.pt'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ckpt_step135000_world.pth'

# закон 13: mmap обязателен -- снимок с состоянием оптимизатора весит
# гигабайты, и грузить его целиком незачем
_sd_train = torch.load(SRC, map_location='cpu', mmap=True, weights_only=False)
_mp = _sd_train['master_params']
_shapes = [tuple(t.shape) for t in _mp]

VOCAB, C = _shapes[0]                       # emb.weight идёт первым
DIM_FFN = next(a for a, b in _shapes if b == C and a % C == 0 and a > C)
# ранги лор: формы (C, r) в первом блоке, в порядке w, a, (v), g
_ranks = [b for a, b in _shapes[:40] if a == C and b < C]
D_DECAY_LORA, D_AAA_LORA = _ranks[0], _ranks[1]
D_MV_LORA = _ranks[2] if len(_ranks) > 3 else _ranks[1]
D_GATE_LORA = _ranks[3] if len(_ranks) > 3 else _ranks[2]
H, N = next((a, b) for a, b in _shapes if a * b == C and a < b)
# слоёв -- по числу матриц att.receptance (их ровно одна на блок)
N_LAYER = sum(1 for sh in _shapes if sh == (C, C)) // 4
print(f"из чекпоинта: C={C} vocab={VOCAB} n_layer={N_LAYER} ffn={DIM_FFN} "
      f"ранги w/a/v/g = {D_DECAY_LORA}/{D_AAA_LORA}/{D_MV_LORA}/{D_GATE_LORA} "
      f"H={H} N={N}")

def expect(name):
    # (name, expected_shape or None if variable/skip strict check)
    if name.endswith(("x_r","x_w","x_k","x_v","x_a","x_g")) and ".att." in name:
        return (1,1,C)
    if name.endswith("w1"): return (C, D_DECAY_LORA)
    if name.endswith("w2"): return (D_DECAY_LORA, C)
    if name.endswith("w0"): return (1,1,C)
    if name.endswith("a1"): return (C, D_AAA_LORA)
    if name.endswith("a2"): return (D_AAA_LORA, C)
    if name.endswith("a0"): return (1,1,C)
    if name.endswith("v1"): return (C, D_MV_LORA)
    if name.endswith("v2"): return (D_MV_LORA, C)
    if name.endswith("v0"): return (1,1,C)
    if name.endswith("g1"): return (C, D_GATE_LORA)
    if name.endswith("g2"): return (D_GATE_LORA, C)
    if name.endswith("k_k") or name.endswith("k_a"): return (1,1,C)
    if name.endswith("r_k"): return (H, N)
    if name.endswith(("receptance.weight","key.weight" if ".att." in name else "___", "value.weight" if ".att." in name else "___","output.weight")):
        pass
    if ".att." in name and name.endswith("receptance.weight"): return (C,C)
    if ".att." in name and name.endswith("key.weight"): return (C,C)
    if ".att." in name and name.endswith("value.weight"): return (C,C)
    if ".att." in name and name.endswith("output.weight"): return (C,C)
    if name.endswith("ln_x.weight") or name.endswith("ln_x.bias"): return (C,)
    if name.endswith("ffn.x_k"): return (1,1,C)
    if name.endswith("ffn.key.weight"): return (DIM_FFN, C)
    if name.endswith("ffn.value.weight"): return (C, DIM_FFN)
    if name in ("emb.weight",): return (65536, C)
    if name in ("ln_out.weight","ln_out.bias"): return (C,)
    if name.endswith("ln0.weight") or name.endswith("ln0.bias"): return (C,)
    if name.endswith("ln1.weight") or name.endswith("ln1.bias"): return (C,)
    if name.endswith("ln2.weight") or name.endswith("ln2.bias"): return (C,)
    return None

def block_names(i):
    names = []
    if i == 0:
        names += [f"blocks.{i}.ln0.weight", f"blocks.{i}.ln0.bias"]
    names += [f"blocks.{i}.ln1.weight", f"blocks.{i}.ln1.bias"]
    names += [f"blocks.{i}.ln2.weight", f"blocks.{i}.ln2.bias"]
    ap = f"blocks.{i}.att."
    names += [ap+n for n in ["x_r","x_w","x_k","x_v","x_a","x_g"]]
    names += [ap+n for n in ["w1","w2","w0"]]
    names += [ap+n for n in ["a1","a2","a0"]]
    if i != 0:
        names += [ap+n for n in ["v1","v2","v0"]]
    names += [ap+n for n in ["g1","g2"]]
    names += [ap+n for n in ["k_k","k_a","r_k"]]
    names += [ap+n for n in ["receptance.weight","key.weight","value.weight","output.weight"]]
    names += [ap+n for n in ["ln_x.weight","ln_x.bias"]]
    fp = f"blocks.{i}.ffn."
    names += [fp+n for n in ["x_k","key.weight","value.weight"]]
    return names

names = ["emb.weight"]
for i in range(N_LAYER):
    names += block_names(i)
names += ["ln_out.weight", "ln_out.bias"]

sd_train, mp = _sd_train, _mp
print("names:", len(names), "master_params:", len(mp))
assert len(mp) == len(names), f"COUNT MISMATCH {len(mp)} vs {len(names)}"

mismatches = []
for idx, (name, tensor) in enumerate(zip(names, mp)):
    exp = expect(name)
    actual = tuple(tensor.shape)
    if exp is not None and tuple(exp) != actual:
        mismatches.append((idx, name, exp, actual))

print(f"mismatches: {len(mismatches)} / {len(names)}")
for m in mismatches[:30]:
    print(m)

if not mismatches:
    out_sd = {name: tensor.clone() for name, tensor in zip(names, mp)}
    out_sd["head.weight"] = torch.zeros(65536, C, dtype=out_sd["emb.weight"].dtype)  # placeholder, unused in act-stats
    torch.save(out_sd, OUT)
    print("SAVED", OUT, "tensors:", len(out_sd))
else:
    print("NOT SAVED — fix mismatches first")
