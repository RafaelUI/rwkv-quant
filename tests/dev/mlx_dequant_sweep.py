import sys, time, collections
sys.path.insert(0, "/Users/s/Develop/rwkv-quant")
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats import reader
sys.path.insert(0, "/Users/s/Develop/rwkv-quant/tests/dev")
from mlx_dequant_proto import dequantize_gw_sb6_mlx, dequantize_gw_asym_mlx, t2mx

def check(path):
    print(f"=== {path} ===")
    ckpt = reader.load_raw(path)
    by_mode_bits = collections.defaultdict(list)
    for k, qt in ckpt.tensors.items():
        by_mode_bits[(qt.gw_mode, qt.bits, qt.gw_qh is not None, qt.gw_qh2 is not None)].append(k)

    for sig, keys in sorted(by_mode_bits.items(), key=lambda kv: str(kv[0])):
        mode, bits, has_qh, has_qh2 = sig
        if mode not in ("sb6", "asym"):
            continue
        k = keys[0]
        qt = ckpt.tensors[k]
        t0 = time.time()
        if mode == "sb6":
            w_ref = reader._dequantize_gw_sb6(qt)
            w_mlx = dequantize_gw_sb6_mlx(qt)
        else:
            w_ref = reader._dequantize_gw_asym(qt)
            w_mlx = dequantize_gw_asym_mlx(qt)
        mx.eval(w_mlx)
        dt = time.time() - t0
        ref_bits = w_ref.view(torch.int16).numpy()
        got_bits = np.array(w_mlx.view(mx.uint16)).astype(np.int16)
        n_mis = int((ref_bits != got_bits).sum())
        print(f"  mode={mode} bits={bits} qh={has_qh} qh2={has_qh2} n_tensors={len(keys):3d} "
              f"sample={k:30s} shape={tuple(qt.shape)} mismatch={n_mis}/{w_ref.numel()} dt={dt*1000:.1f}ms")

check("/tmp/reduction_v2.rwkvq")
check("/tmp/champion_v2.rwkvq")
