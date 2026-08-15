import torch

SRC = '/Users/s/Develop/ckpt_step135000.pt'
OUT = '/tmp/ckpt_step135000_world.pth'
N_LAYER = 24

def block_names(i):
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

names = ["emb.weight"]
for i in range(N_LAYER):
    names += block_names(i)
names += ["ln_out.weight", "ln_out.bias"]

sd_train = torch.load(SRC, map_location='cpu')
mp = sd_train['master_params']
assert len(mp) == len(names), f"{len(mp)} vs {len(names)}"

out_sd = {name: tensor.clone() for name, tensor in zip(names, mp)}
# head.weight структурно требуется загрузчиком RWKV7Ref, но не используется
# в activation-stats (см. _tmix_forward/_cmix_forward — head туда не входит).
# Модель эмбеддингов реально головы не имеет, поэтому тай на emb.weight —
# это placeholder, а не заявление о реальной архитектуре.
out_sd["head.weight"] = mp[0].clone()

torch.save(out_sd, OUT)
print("saved", OUT, "tensors:", len(out_sd))
print("emb.weight", tuple(out_sd["emb.weight"].shape))
print("blocks.0.att.w1", tuple(out_sd["blocks.0.att.w1"].shape))
print("blocks.1.att.v1", tuple(out_sd["blocks.1.att.v1"].shape))
