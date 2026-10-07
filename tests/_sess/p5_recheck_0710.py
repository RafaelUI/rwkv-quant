"""07.10: перепроверка находки 5 (04.10) на нынешнем коде -- потоковый префилл (128 разом + 128 по одному) против цельного
(256 разом): max|dlogit| на последней позиции (число из api_smoke_0410) и KL по позициям 128..255. Плюс пол: 128 + 128.
    python p5_recheck_0710.py <файл.rwkvq> ..."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, mlx.core as mx
from rwkv_quant.formats.reader import load_raw
from rwkv_quant.backends.metal import quant_model as qm
ev = torch.load(os.path.expanduser("~/Develop/WKV-kvant/eval_text_heldout.pt"), weights_only=False)["tokens"]
WINS = {"en": ev[0, :256], "ru": ev[12, :256], "sr": ev[24, :256]}
def lsm(a): return torch.log_softmax(torch.from_numpy(a), -1)
def kl(p, q): return float((p.exp() * (p - q)).sum(-1).mean())
def run(m, ids, splits):
    st = m.init_state(1); outs = []; p = 0
    for s in splits:
        lg, st = m.step(ids[:, p:p + s], st); lg = lg.astype(mx.float32); mx.eval(lg, st); outs.append(np.array(lg)[0]); p += s
    return np.concatenate(outs, 0)
for f in sys.argv[1:]:
    raw = load_raw(f); m = qm.QuantRWKV7(raw)
    print("%s: LORA_Q модуля %r, решение модели lora_q=%r" % (os.path.basename(f), qm.LORA_Q, getattr(m, "lora_q", "?")))
    for n, w in WINS.items():
        ids = mx.array(w.numpy().astype(np.int32)[None])
        whole = run(m, ids, [256]); half = run(m, ids, [128, 128]); strm = run(m, ids, [128] + [1] * 128)
        print("   %s  последняя позиция max|dlogit|: поток %.4f, нарезка 128+128 %.4f | KL 128..255: поток %.2e, нарезка %.2e | argmax совпал на %d / 128"
              % (n, np.abs(strm[-1] - whole[-1]).max(), np.abs(half[-1] - whole[-1]).max(), kl(lsm(whole[128:]), lsm(strm[128:])),
                 kl(lsm(whole[128:]), lsm(half[128:])), int((strm[128:].argmax(-1) == whole[128:].argmax(-1)).sum())), flush=True)
    del m; mx.clear_cache()
