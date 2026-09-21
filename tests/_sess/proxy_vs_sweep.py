import json,glob,sys
import numpy as np
from scipy.stats import spearmanr
s={}
for f in glob.glob(sys.argv[1]): s.update({k:v["kl"] for k,v in json.load(open(f)).items() if k!="_meta" and "kl" in v and not k.startswith("__")})
p=json.load(open(sys.argv[2]))["rows"]
ks=[k for k in s if k in p]
a=np.array([s[k] for k in ks]); b=np.array([p[k]["score"] for k in ks])
rho=spearmanr(a,b).correlation; lr=np.corrcoef(np.log(a),np.log(b))[0,1]
print("плеч %d  Spearman %.3f  corr(log) %.3f  медиана score/KL %.2f"%(len(ks),rho,lr,np.median(b/a)))
ts=[ks[i] for i in np.argsort(-a)]; tp=[ks[i] for i in np.argsort(-b)]
for n in (3,5,10,20): print("  верх-%d: общих %d"%(n,len(set(ts[:n])&set(tp[:n]))))
print("  истина -> место у признака, отношение score/KL:")
for k in ts[:10]: print("    %-32s KL %.6f  место %3d  %.2f"%(k,s[k],tp.index(k)+1,p[k]["score"]/s[k]))
print("  признак-верх, которого нет в истинном верх-10:")
for k in tp[:10]:
    if k not in ts[:10]: print("    %-32s score %.6f  истинное место %3d"%(k,p[k]["score"],ts.index(k)+1))
