"""Переписать safetensors-заголовок .rwkvq с пробелами так, чтобы данные
начинались с заданного смещения по модулю M (данные и JSON не меняются).
    python pad_header_2109.py <вход> <выход> <M> <остаток>"""
import json, struct, sys, shutil
src, dst, M, R = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = f.read(n)
    json.loads(hdr)
    start = 8 + n
    pad = (R - start) % M
    with open(dst, "wb") as g:
        g.write(struct.pack("<Q", n + pad)); g.write(hdr + b" " * pad)
        shutil.copyfileobj(f, g, 1 << 24)
print("заголовок %d -> %d, данные с %d (mod %d = %d)" % (n, n + pad, start + pad, M, (start + pad) % M))
