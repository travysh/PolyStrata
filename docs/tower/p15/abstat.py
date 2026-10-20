"""Pass 15: summarises p15ab.sh output (p14/abstat.py plus the CPU busy share and the batches with sleepers).  Per run: the short requests' and each turn's decode tok/s, and the engine's
CPU-expert ms per generated token over the whole run (its "PASS14 request" lines); then the mean per label and dir."""
import collections, re, sys

runs, cur = [], None
for l in open(sys.argv[1]):
    m = re.match(r'== (\S+) (\w+) (\d+) ', l)
    if m:
        cur = {'v': m[1], 'd': m[2], 'short': [], 't': {}, 'ok': None, 'cpu_ms': 0.0, 'gen': 0, 'loaded': '', 'busy': None, 'sl': []}
        cur['loaded'] = (m2[1] + 'us') if (m2 := re.search(r'spin (\d+) us', l)) else ''
        runs.append(cur)
        continue
    if cur is None:
        continue
    if m := re.match(r'short \d.*decode ([\d.]+)', l):
        cur['short'].append(float(m[1]))
    elif m := re.match(r'\s+(\d)\s+(\d+)\s+(\d+)\s+[\d.]+\s+(\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)', l):
        cur['t'][int(m[1])] = (int(m[2]), float(m[5]))
    elif l.strip() in ('PASS', 'FAIL'):
        cur['ok'] = l.strip()
    elif m := re.search(r'PASS14 request: (\d+) generated, CPU experts ([\d.]+) ms', l):
        cur['gen'] += int(m[1]); cur['cpu_ms'] += float(m[2])
    elif m := re.search(r'CPU busy during multiturn: ([\d.]+)%', l):
        cur['busy'] = float(m[1])
    elif m := re.search(r'PASS15 workers: (\d+) batches, ([\d.]+)% with sleepers', l):
        cur['sl'].append((int(m[1]), float(m[2])))

def sleep(r):
    return sum(n * p for n, p in r['sl']) / max(sum(n for n, _ in r['sl']), 1)

for r in runs:
    print(f"{r['v']:7s} {r['d']} {r['ok']} log:{r['loaded'] or '-':6s} short {r['short']} decode "
          f"{[r['t'][k][1] for k in sorted(r['t'])]} ctx {[r['t'][k][0] // 1000 for k in sorted(r['t'])]}K "
          f"CPU experts {r['cpu_ms'] / max(r['gen'], 1):.2f} ms/token, busy {r['busy']}%, sleepers {sleep(r):.1f}%")
groups = collections.defaultdict(list)
for r in runs:
    groups[(r['v'].replace('w1-', ''), r['d'], r['v'].startswith('w1'))].append(r)
print("means:")
for k in sorted(groups):
    rs = groups[k]; turns = sorted(set().union(*(r['t'] for r in rs)))
    dec = [round(sum(r['t'][t][1] for r in rs if t in r['t']) / sum(1 for r in rs if t in r['t']), 1) for t in turns]
    sh = [x for r in rs for x in r['short']]
    cpu = sum(r['cpu_ms'] for r in rs) / max(sum(r['gen'] for r in rs), 1)
    print(f"  {'window 1 ' if k[2] else ''}{k[0]:4s} {k[1]} (n={len(rs)}): short {sum(sh) / len(sh):.1f}  decode {dec}  "
          f"CPU experts {cpu:.2f} ms/token, busy {sum(r['busy'] or 0 for r in rs) / len(rs):.1f}%, "
          f"sleepers {sum(sleep(r) for r in rs) / len(rs):.1f}%")
