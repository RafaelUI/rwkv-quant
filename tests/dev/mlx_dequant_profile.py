import sys, time
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests/dev")
import mlx.core as mx
from rwkv_quant.formats import reader
from mlx_dequant_proto import dequantize_gw_sb6_mlx

ckpt = reader.load_raw("/tmp/reduction_v2.rwkvq")
qt = ckpt.tensors["emb.weight"]
print("shape", qt.shape)

for i in range(5):
    t0 = time.time()
    w = dequantize_gw_sb6_mlx(qt)
    mx.eval(w)
    print(f"run {i}: {(time.time()-t0)*1000:.1f}ms")

# профиль по шагам (последний прогон, тёплый)
import mlx.core as mx
from rwkv_quant.formats.reader import _dequantize_gw_sb6 as _ref
from mlx_dequant_proto import t2mx, unpack_nib_block_mlx, unpack_bitplane_mlx, unpack6_mlx

OUT, IN = qt.shape
gs, NB = qt.gw_gs, IN // qt.gw_gs

t0 = time.time(); codes_packed = t2mx(qt.codes_packed); mx.eval(codes_packed); print("t2mx codes_packed", (time.time()-t0)*1000)
t0 = time.time(); q = unpack_nib_block_mlx(codes_packed, gs).astype(mx.float32); mx.eval(q); print("unpack_nib_block", (time.time()-t0)*1000)
t0 = time.time(); qh = t2mx(qt.gw_qh); q2 = q + unpack_bitplane_mlx(qh, IN).astype(mx.float32)*16.0; mx.eval(q2); print("qh", (time.time()-t0)*1000)
t0 = time.time(); qh2 = t2mx(qt.gw_qh2); q3 = q2 + unpack_bitplane_mlx(qh2, IN).astype(mx.float32)*32.0; mx.eval(q3); print("qh2", (time.time()-t0)*1000)
t0 = time.time(); qsqm = t2mx(qt.gw_qsqm); mx.eval(qsqm); print("t2mx qsqm", (time.time()-t0)*1000)
t0 = time.time(); qs = unpack6_mlx(qsqm[..., :6], 8).reshape(OUT, NB).astype(mx.float32); mx.eval(qs); print("unpack6 qs", (time.time()-t0)*1000)
t0 = time.time(); qm = unpack6_mlx(qsqm[..., 6:], 8).reshape(OUT, NB).astype(mx.int32).astype(mx.float32) - 31.0; mx.eval(qm); print("unpack6 qm", (time.time()-t0)*1000)
t0 = time.time(); d = t2mx(qt.gw_d).astype(mx.float32); dm = t2mx(qt.gw_dm).astype(mx.float32); mx.eval(d, dm); print("t2mx d/dm", (time.time()-t0)*1000)
t0 = time.time(); d_c = mx.repeat(d, qt.gw_sb, axis=1); dm_c = mx.repeat(dm, qt.gw_sb, axis=1); mx.eval(d_c, dm_c); print("repeat d/dm->NB", (time.time()-t0)*1000)
t0 = time.time(); scale = (qs*d_c).astype(mx.float16).astype(mx.float32); scale = mx.maximum(scale, 1e-8); mn = (qm*dm_c).astype(mx.float16).astype(mx.float32); mx.eval(scale, mn); print("scale/mn", (time.time()-t0)*1000)
t0 = time.time(); scale_c = mx.repeat(scale, gs, axis=1); mn_c = mx.repeat(mn, gs, axis=1); mx.eval(scale_c, mn_c); print("repeat scale/mn -> IN", (time.time()-t0)*1000)
t0 = time.time(); w = (q3*scale_c + mn_c).astype(mx.bfloat16); mx.eval(w); print("final mul+add+cast", (time.time()-t0)*1000)
