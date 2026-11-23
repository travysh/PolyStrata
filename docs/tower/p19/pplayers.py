# pplayers.py LOG: per request (PP lines between ms resets), each line's layer span and ms per layer
import sys
reqs, cur, last = [], [], None
for ln in open(sys.argv[1]):
    t, _, done, total, ms, *_ = ln.split()
    done, total, ms = int(done), int(total), float(ms)
    if last and (ms < last[2] or total != last[1]): reqs.append(cur); cur = []
    cur.append((done, total, ms)); last = (done, total, ms)
reqs.append(cur)
for r in reqs:
    total = r[0][1]; per = total / 41
    print(f"== {total} tokens, {len(r)} lines, last {r[-1][2]:.0f} ms")
    prev = (0, 0.0)
    rows = []
    for done, _, ms in r:
        l0, l1 = prev[0] / per, done / per
        rows.append(f"L{l0:5.1f}-{l1:5.1f} {(ms - prev[1]) / max(l1 - l0, 1e-9):7.0f} ms/layer")
        prev = (done, ms)
    print("\n".join(rows))
