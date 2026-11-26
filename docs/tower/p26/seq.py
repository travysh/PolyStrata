#!/usr/bin/env python3
"""p26/seq.py TRACE.csv.gz K: the K-th last decode token's GPU work in order (start offset, duration, gap before, name)
and the main thread's runtime calls longer than 5 us interleaved."""
import gzip, sys, re
def short(n): return n[:200]
rows = []
with (gzip.open if sys.argv[1].endswith(".gz") else open)(sys.argv[1], "rt", errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split(",", 5)
        if len(p) < 6: continue
        if p[0] in "KMS": rows.append((int(p[1]), int(p[2]), "G", p[0] + ":" + (short(p[5]) if p[0] == "K" else p[5][:20])))
        elif p[0] == "R": rows.append((int(p[1]), int(p[2]), "H" + p[3], p[5][:30]))
import collections
main = "H" + collections.Counter(r[2] for r in rows if r[2] != "G").most_common(1)[0][0][1:]
raw = sorted(r[1] for r in rows if r[2] == main and r[3].startswith("cudaStreamSynchronize"))
syncs = [t for i, t in enumerate(raw) if i == 0 or t - raw[i - 1] > 1e6]
k = int(sys.argv[2]); t0, t1 = syncs[-k - 1], syncs[-k]
sel = sorted(r for r in rows if t0 <= r[0] < t1 and (r[2] == "G" or (r[2] == main and r[1] - r[0] > 5000)))
last = t0
for s, e, w, nm in sel:
    if w == "G":
        print(f"{(s - t0) / 1e3:8.1f} {(e - s) / 1e3:7.1f} gap {(s - last) / 1e3:6.1f}  {nm}"); last = max(last, e)
    else:
        print(f"{(s - t0) / 1e3:8.1f} {(e - s) / 1e3:7.1f}      host  {nm}")
