#!/usr/bin/env python3
"""p26/tr.py TRACE.csv[.gz] [TOKENS]: cupti_trace.so's CSV, the last TOKENS (default 100) decode tokens.
A token ends at the main thread's cudaStreamSynchronize; prints a token's wall time, the GPU's busy time and its
idle gaps, the kernels by time, and the host's runtime calls by time."""
import gzip, sys, collections, re

def short(name):
    name = re.sub(r"\(.*", "", name)                # drop the argument list
    name = re.sub(r"<.*", "", name)                  # and template arguments
    return name.split("::")[-1][:60]

path = sys.argv[1]; ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 100
op = gzip.open if path.endswith(".gz") else open
gpu, api = [], []
with op(path, "rt", errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split(",", 5)
        if len(p) < 6: continue
        k = p[0]
        if k == "K": gpu.append((int(p[1]), int(p[2]), int(p[3]), "K:" + short(p[5])))
        elif k == "M":
            q = p[5].split(","); gpu.append((int(p[1]), int(p[2]), int(p[3]), "M:kind%s" % q[0]))
        elif k == "S": gpu.append((int(p[1]), int(p[2]), int(p[3]), "S:memset"))
        elif k == "R": api.append((int(p[1]), int(p[2]), int(p[3]), p[5]))
main = collections.Counter(a[2] for a in api).most_common(1)[0][0]
raw = sorted(a[1] for a in api if a[2] == main and a[3].startswith("cudaStreamSynchronize"))
syncs = [t for i, t in enumerate(raw) if i == 0 or t - raw[i - 1] > 1e6]   # a token's syncs come in a cluster
# decode tokens: the last ntok+1 syncs that are 5-200 ms apart
ends = []
for t in reversed(syncs):
    if ends and not (2e6 < ends[-1] - t < 2e8): break
    ends.append(t)
ends = sorted(ends)[-(ntok + 1):]
t0, t1 = ends[0], ends[-1]; n = len(ends) - 1
wall = (t1 - t0) / n / 1e3
print(f"tokens {n}, {wall:.0f} us a token ({1e6 / wall:.1f} tok/s traced)")
g = sorted(x for x in gpu if x[1] > t0 and x[0] < t1)
busy = 0; cur_s = cur_e = None; gaps = []
for s, e, st, nm in g:
    s = max(s, t0); e = min(e, t1)
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s; gaps.append((s - cur_e, cur_e))
        cur_s, cur_e = s, e
    else: cur_e = max(cur_e, e)
busy += cur_e - cur_s
print(f"GPU busy {busy / n / 1e3:.0f} us a token ({100 * busy / (t1 - t0):.0f}%), idle {(t1 - t0 - busy) / n / 1e3:.0f} us")
gl = sorted(x[0] for x in gaps)
hist = collections.Counter()
for d in gl:
    b = "<2us" if d < 2e3 else "2-5us" if d < 5e3 else "5-20us" if d < 2e4 else "20-50us" if d < 5e4 else "50-200us" if d < 2e5 else ">200us"
    hist[b] += d
print("idle by gap length (us a token):", ", ".join(f"{b} {hist[b] / n / 1e3:.0f}" for b in ["<2us", "2-5us", "5-20us", "20-50us", "50-200us", ">200us"]))
print("gaps a token:", len(gl) / n)
ks = collections.defaultdict(lambda: [0, 0])
for s, e, st, nm in g:
    ks[nm][0] += e - s; ks[nm][1] += 1
tot = sum(v[0] for v in ks.values())
print(f"\nkernels/copies: {tot / n / 1e3:.0f} us a token of device time; top:")
for nm, (t, c) in sorted(ks.items(), key=lambda x: -x[1][0])[:30]:
    print(f"  {t / n / 1e3:7.1f} us  {c / n:6.1f}/tok  {t / c / 1e3:6.2f} us each  {nm}")
streams = collections.defaultdict(int)
for s, e, st, nm in g: streams[st] += e - s
print("by stream:", {k: round(v / n / 1e3) for k, v in streams.items()})
aa = collections.defaultdict(lambda: [0, 0])
for s, e, th, nm in api:
    if e > t0 and s < t1: aa[(th == main, nm)][0] += min(e, t1) - max(s, t0); aa[(th == main, nm)][1] += 1
print("\nhost runtime calls (main thread first):")
for (m, nm), (t, c) in sorted(aa.items(), key=lambda x: (not x[0][0], -x[1][0]))[:24]:
    print(f"  {'main ' if m else 'other'} {t / n / 1e3:7.1f} us  {c / n:7.1f}/tok  {nm}")
