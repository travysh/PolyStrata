"""Pass 14: summarises p14bench.sh output: per variant the mean of the rounds, overall and per call shape, and the
change against the first variant named (today; the mean of its runs in that round), compared within each round."""
import re, sys, collections
rows = collections.defaultdict(lambda: collections.defaultdict(list)); hashes = collections.defaultdict(set); order = []
for l in open(sys.argv[1]):
    m = re.match(r'round (\d+) (\S+)\s+trace .*?: ([\d.]+) ms .*hash (\w+) \|(.*)', l)
    if not m: continue
    r, v, tot, h, rest = int(m[1]), m[2], float(m[3]), m[4], m[5]
    if v not in order: order.append(v)
    rows[v]['all'].append((r, tot)); hashes[v].add(h)
    for name, ms in re.findall(r' ([\w -]+?): \d+ calls ([\d.]+) ms', rest): rows[v][name].append((r, float(ms)))
base = order[0]
keys = list(rows[base].keys())
print(f"{'variant':10s}" + "".join(f"{k:>24s}" for k in keys) + "  hash")
for v in order:
    cells = []
    for k in keys:
        b = collections.defaultdict(list)
        for r, x in rows[base][k]: b[r].append(x)
        b = {r: sum(x) / len(x) for r, x in b.items()}   # today's mean in that round (it may run more than once)
        vals = rows[v][k]
        mean = sum(x for _, x in vals) / len(vals)
        ch = [100 * (x / b[r] - 1) for r, x in vals if r in b]
        cells.append(f"{mean:9.1f} ({min(ch):+5.1f}..{max(ch):+5.1f}%)")
    print(f"{v:10s}" + "".join(f"{c:>24s}" for c in cells) + "  " + ",".join(hashes[v]))
