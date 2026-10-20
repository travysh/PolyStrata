"""Pass 14: the shape of the engine's CPU expert calls in a STRATA_CPU_TRACE file (one line per call:
layer tokens uses us launch expert:token ...)."""
import sys, collections
by_m = collections.defaultdict(lambda: [0, 0.0]); by_run = collections.defaultdict(lambda: [0, 0.0])
uses_per = collections.Counter(); layers = collections.Counter(); tot = n = launch = bad = 0
for l in open(sys.argv[1]):
    f = l.split()
    try:
        il, m, nh, us, la = int(f[0]), int(f[1]), int(f[2]), float(f[3]), int(f[4])
        es = [int(x.split(':')[0]) for x in f[5:]]
        assert len(es) == nh
    except (ValueError, IndexError, AssertionError):
        bad += 1
        continue
    c = collections.Counter(es); runs = len(c)
    for v in c.values(): uses_per[v] += 1
    by_m[m][0] += 1; by_m[m][1] += us
    b = '01-04' if runs <= 4 else '05-08' if runs <= 8 else '09-16' if runs <= 16 else '17+'
    k = ('1 token' if m == 1 else '2+ tokens', b)
    by_run[k][0] += 1; by_run[k][1] += us
    tot += us; n += 1; layers[il] += 1; launch += la
print(f"{n} calls ({bad} unreadable lines), {tot/1e6:.1f} s in cpu_experts, layers {min(layers)}-{max(layers)}, "
      f"GPU launches run meanwhile in {launch/n:.0%}")
print("tokens a call: calls, share of calls, share of time, us a call")
for m in sorted(by_m):
    c, t = by_m[m]; print(f"  {m}: {c} {c/n:.1%} {t/tot:.1%} {t/c:.0f}")
print("by tokens and distinct experts: calls, share of time, us a call")
for k in sorted(by_run):
    c, t = by_run[k]; print(f"  {k}: {c} {t/tot:.1%} {t/c:.0f}")
s = sum(uses_per.values())
print("uses an expert has in a call:", {k: f"{v/s:.1%}" for k, v in sorted(uses_per.items())[:8]})
