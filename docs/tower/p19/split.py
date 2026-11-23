# split.py LOG...: per 64K read, mean ms of a full-attention layer (3, 7, ... 39) and of a GDN layer, from the PP
# lines that span exactly one layer (41 turns: 40 layers and the draft block); layer 0 and the last turn left out
import sys
def reqs(p):
    rs, cur, last = [], [], None
    for ln in open(p):
        _, _, d, tot, ms, *_ = ln.split(); d, tot, ms = int(d), int(tot), float(ms)
        if last and (ms < last[2] or tot != last[1]): rs.append(cur); cur = []
        cur.append((d, tot, ms)); last = (d, tot, ms)
    rs.append(cur); return rs
for p in sys.argv[1:]:
    for r in reqs(p):
        tot = r[0][1]
        if tot < 60000 or len(r) < 30: continue
        per = tot / 41; att, gdn = [], []; prev = (0, 0.0)
        for d, _, ms in r:
            l0, l1 = prev[0] / per, d / per
            if abs(l1 - l0 - 1) < 0.05 and 0.95 <= l0 < 39.5:
                (att if round(l0) % 4 == 3 else gdn).append(ms - prev[1])
            prev = (d, ms)
        m = lambda v: f"{sum(v)/len(v):5.0f} x{len(v):2d}" if v else "   -"
        print(f"{p:24s} {r[-1][2]:6.0f} ms  attention layer {m(att)}  GDN layer {m(gdn)}")
