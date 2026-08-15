import torch

sd = torch.load('/Users/s/Develop/ckpt_step135000.pt', map_location='cpu')
mp = sd['master_params']
print("master_params len:", len(mp))

N_LAYER = 24

def block_names(i, n_layer):
    names = []
    if i == 0:
        names += [f"blocks.{i}.ln0.weight", f"blocks.{i}.ln0.bias"]
    names += [f"blocks.{i}.ln1.weight", f"blocks.{i}.ln1.bias"]
    ap = f"blocks.{i}.att."
    names += [ap+n for n in ["x_r","x_w","x_k","x_v","x_a","x_g"]]
    names += [ap+n for n in ["w0","w1","w2","a0","a1","a2"]]
    if i != 0:
        names += [ap+n for n in ["v0","v1","v2"]]
    names += [ap+n for n in ["g1","g2","k_k","k_a","r_k"]]
    names += [ap+n for n in ["receptance.weight","key.weight","value.weight","output.weight"]]
    names += [ap+n for n in ["ln_x.weight","ln_x.bias"]]
    names += [f"blocks.{i}.ln2.weight", f"blocks.{i}.ln2.bias"]
    fp = f"blocks.{i}.ffn."
    names += [fp+n for n in ["x_k","key.weight","value.weight"]]
    return names

for with_head in (False, True):
    names = ["emb.weight"]
    for i in range(N_LAYER):
        names += block_names(i, N_LAYER)
    names += ["ln_out.weight", "ln_out.bias"]
    if with_head:
        names.append("head.weight")
    print(f"with_head={with_head} -> {len(names)} names (target 794)")

# use the variant matching 794 (or closest) and show shape zip check
names = ["emb.weight"]
for i in range(N_LAYER):
    names += block_names(i, N_LAYER)
names += ["ln_out.weight", "ln_out.bias"]
print("no-head total:", len(names))
diff = len(mp) - len(names)
print("diff (master_params - names):", diff)

# print first/last few shapes for sanity
for idx in list(range(6)) + list(range(len(names)-6, len(names))):
    nm = names[idx] if idx < len(names) else "???"
    sh = tuple(mp[idx].shape) if idx < len(mp) else "???"
    print(idx, nm, sh)
