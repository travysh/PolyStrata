"""Pass 17: summarises p17bench.sh output: per variant the mean over rounds of the slice's time, and of each phase's
wall time and idle share by call shape (all shapes weighted by wall time), the hashes, then design decision 3's rule.
    python3 p17stat.py /data/pass17-split.txt"""
import re, sys, collections
tot = collections.defaultdict(list); hashes = collections.defaultdict(set); order = []
ph = collections.defaultdict(lambda: collections.defaultdict(list))   # variant -> (shape, phase) -> [(wall, idle)]
v = None
for l in open(sys.argv[1]):
    m = re.match(r'round \d+ (\S+)\s+trace .*?: ([\d.]+) ms .*hash (\w+)', l)
    if m:
        v = m[1]; tot[v].append(float(m[2])); hashes[v].add(m[3])
        if v not in order: order.append(v)
        continue
    m = re.match(r'\s+(1 token|2 tokens|3-8 tokens)\s+call\s+[\d.]+ ms:(.*)', l)
    if m and v:
        for name, wall, idle in re.findall(r' (act|gate/up|h|down) ([\d.]+) ms idle ([\d.]+)%', m[2]):
            ph[v][(m[1], name)].append((float(wall), float(idle)))
mean = lambda xs: sum(xs) / len(xs)
shapes = ['1 token', '2 tokens', '3-8 tokens']; phases = ['act', 'gate/up', 'h', 'down']

def phase(v, p, shape=None):   # mean wall ms over rounds, summed over shapes; idle weighted by wall
    w = i = 0.0
    for s in ([shape] if shape else shapes):
        xs = ph[v][(s, p)]
        if not xs: continue
        ww = mean([a for a, _ in xs]); w += ww; i += ww * mean([b for _, b in xs])
    return w, (i / w if w else 0.0)

base = order[0]
print(f"{'variant':8s} {'slice ms':>9s} {'vs ' + base:>8s}  " + "  ".join(f"{p + ' ms (idle)':>20s}" for p in phases) + "  rounds  hash")
for v in order:
    cells = [f"{phase(v, p)[0]:10.1f} ({phase(v, p)[1]:4.1f}%)" for p in phases]
    print(f"{v:8s} {mean(tot[v]):9.1f} {100 * (mean(tot[v]) / mean(tot[base]) - 1):+7.1f}%  " + "  ".join(f"{c:>20s}" for c in cells)
          + f"  {len(tot[v])}  {','.join(sorted(hashes[v]))}")
for s in shapes:
    print(f"  full, {s:10s}: " + "  ".join(f"{p} {phase('full', p, s)[0]:.1f} ms idle {phase('full', p, s)[1]:.1f}%" for p in phases))

if 'full' in tot and 'arith' in tot:
    walls = {p: phase('full', p) for p in phases}
    all_wall = sum(w for w, _ in walls.values())
    idle = sum(w * i for w, i in walls.values()) / all_wall
    down_share = walls['down'][0] / mean(tot['full'])
    down_arith = phase('arith', 'down')[0] / walls['down'][0]
    a = idle >= 10; b = down_arith >= 0.70 and down_share >= 0.30
    print(f"\nrule: idle share of full {idle:.1f}% (>= 10%: {a}); down arith {100 * down_arith:.0f}% of full's down"
          f" (>= 70%) and down {100 * down_share:.0f}% of the slice (>= 30%): {b}")
    print("outcome:", "both: dataflow first (group 3)" if a and b else "dataflow (group 3)" if a else
          "down product (group 4)" if b else "neither: write up and stop")
