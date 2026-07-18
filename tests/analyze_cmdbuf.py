import sys, re, statistics

def build_id_map(content):
    m = {}
    for match in re.finditer(r'id="(\d+)"[^>]*>(\d+)<', content):
        m[int(match.group(1))] = int(match.group(2))
    return m

def resolve_first(chunk, tag, id_map):
    m_def = re.search(rf'<{tag}\s+id="(\d+)"[^>]*>(\d+)</{tag}>', chunk)
    m_ref = re.search(rf'<{tag}\s+ref="(\d+)"\s*/>', chunk)
    candidates = []
    if m_def:
        candidates.append((m_def.start(), int(m_def.group(2))))
    if m_ref:
        rid = int(m_ref.group(1))
        if rid in id_map:
            candidates.append((m_ref.start(), id_map[rid]))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]

def parse(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    id_map = build_id_map(content)
    parts = content.split("<row>")[1:]
    rows = []
    for r in parts:
        row_end = r.find("</row>")
        chunk = r[:row_end] if row_end != -1 else r
        st = resolve_first(chunk, "start-time", id_map)
        du = resolve_first(chunk, "duration", id_map)
        if st is not None and du is not None:
            rows.append((st, du))
    return rows

def analyze(path, label):
    rows = parse(path)
    total_parsed = len(rows)
    rows.sort(key=lambda x: x[0])
    med = statistics.median(d for _, d in rows)
    dropped = 0
    while rows and rows[-1][1] > 50 * med:
        rows.pop()
        dropped += 1
    n = len(rows)
    window_ns = (rows[-1][0] + rows[-1][1]) - rows[0][0]
    naive_sum = sum(d for _, d in rows)
    merged = []
    cur_s, cur_e = rows[0][0], rows[0][0] + rows[0][1]
    for s, d in rows[1:]:
        e = s + d
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    occupied_ns = sum(e - s for s, e in merged)
    idle_ns = window_ns - occupied_ns
    print(f"=== {label} ===")
    print(f"parsed rows: {total_parsed}, dropped as flush-artifact: {dropped}, kept: {n}")
    print(f"window: {window_ns/1e9:.4f}s")
    print(f"TRUE occupied (union of buffer spans): {occupied_ns/1e9:.4f}s ({occupied_ns/window_ns*100:.2f}%)")
    print(f"TRUE idle: {idle_ns/1e9:.4f}s ({idle_ns/window_ns*100:.2f}%)")
    print(f"naive sum(Duration) = {naive_sum/1e9:.4f}s (concurrency factor {naive_sum/occupied_ns:.3f}x)")
    print(f"buffers/sec = {n/window_ns*1e9:.0f}, avg Duration/buf = {naive_sum/n/1000:.2f}us, "
          f"median = {med/1000:.2f}us")
    return dict(n=n, window_ns=window_ns, occupied_ns=occupied_ns, idle_ns=idle_ns, naive_sum=naive_sum)

if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
