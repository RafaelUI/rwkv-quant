import json,glob,sys,collections
r={}
for f in sorted(glob.glob(sys.argv[1])): r.update({k:v for k,v in json.load(open(f)).items() if k!="_meta"})
arms=sorted(((v["kl"],k) for k,v in r.items() if "kl" in v and not k.startswith("__")), reverse=True)
tot=sum(k for k,_ in arms)
print("плеч",len(arms),"сумма одиночных %.6f"%tot, " __all__", r.get("__all__",{}).get("kl"), " __none__", r.get("__none__",{}).get("kl"))
for kl,k in arms[:int(sys.argv[2]) if len(sys.argv)>2 else 10]: print("  %.6f  %4.1f%%  %s"%(kl,100*kl/tot,k))
g=collections.defaultdict(float)
for kl,k in arms:
    t="emb" if k=="emb.weight" else "head" if k=="head.weight" else "lora" if k.endswith(".lora") else k.split(".",2)[2].replace(".weight","")
    g[t]+=kl
print("по типам:", " ".join("%s %.0f%%"%(t,100*v/tot) for t,v in sorted(g.items(),key=lambda x:-x[1])))
print("медиана хвоста %.6f"%sorted(k for k,_ in arms)[len(arms)//2])
