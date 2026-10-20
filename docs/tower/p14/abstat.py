"""Pass 14: summarises p14ab.sh output.  Per run: the short requests' and each turn's decode tok/s, and the engine's
CPU-expert ms per generated token over the whole run (its "PASS14 request" lines); then the mean per label and dir."""
import collections, re, sys

runs, cur = [], None
for l in open(sys.argv[1]):
    m = re.match(r'== (\S+) (\w+) (\d+) ', l)
    if m:
        cur = {'v': m[1], 'd': m[2], 'short': [], 't': {}, 'ok': None, 'cpu_ms': 0.0, 'gen': 0, 'loaded': ''}
        cur['loaded'] = 'DOWN_ORDER' if 'STRATA_CPU_DOWN_ORDER' in l else ''
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

for r in runs:
    print(f"{r['v']:7s} {r['d']} {r['ok']} log:{r['loaded'] or '-':10s} short {r['short']} decode "
          f"{[r['t'][k][1] for k in sorted(r['t'])]} ctx {[r['t'][k][0] // 1000 for k in sorted(r['t'])]}K "
          f"CPU experts {r['cpu_ms'] / max(r['gen'], 1):.2f} ms/token")
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
          f"CPU experts {cpu:.2f} ms/token")
