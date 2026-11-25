"""Pass 22, thinking: generated puzzles with one exact answer, asked three ways (thinking off, 2K budget, 12K budget).

Six families (logic grid, scheduling, shortest path, dates and time zones, word problems, unit conversions), three
levels each, two puzzles per level, all made from fixed seeds here, so no model has seen them.  A program computes each
answer and a second method checks it (`selftest`).  Only the reply's last `ANSWER:` line counts.
    python3 think.py selftest            every answer checked two ways, every grader accepts the right answer
    python3 think.py prompts OUT         OUT/<id>/task.json for run_bench.py (id = puzzle-mode, e.g. grid3a-big)
    python3 think.py grade OUT RUNS...   score answers (RUNS.jsonl from run_bench.py); writes RUNS.graded.jsonl
    python3 think.py score GRADED...     the tables
"""
import itertools, json, math, random, re, sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

MODES = {"off": {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 4096},
         "2k": {"reasoning_budget_tokens": 2048, "max_tokens": 4096},
         "big": {"reasoning_budget_tokens": 12288, "max_tokens": 16384}}
TAIL = "\n\nWork it out, then end your reply with one last line of the form\nANSWER: {fmt}"


# ---------------------------------------------------------------- logic grid
NAMES = ["Ava", "Ben", "Cleo", "Dev", "Ines"]
VALUES = {"color": ["red", "green", "blue", "white", "yellow"], "pet": ["cat", "dog", "fish", "parrot", "rabbit"],
          "drink": ["tea", "coffee", "milk", "juice", "water"]}


def grid_text(c, v):
    return {"name": v, "color": f"the person in the {v} house", "pet": f"the {v} owner", "drink": f"the {v} drinker"}[c]


def clue_ok(clue, pos):
    t, a, b = clue
    pa = pos[a]
    if t == "at":
        return pa == b
    pb = pos[b]
    return {"same": pa == pb, "not": pa != pb, "left": pa + 1 == pb, "next": abs(pa - pb) == 1, "before": pa < pb}[t]


def clue_cats(clue):
    return {clue[1][0]} | ({clue[2][0]} if clue[0] != "at" else set())


def grid_solutions(n, cats, vals, clues, limit=2):
    found = []

    def go(i, assign):
        if len(found) >= limit:
            return
        if i == len(cats):
            found.append(dict(assign))
            return
        c = cats[i]
        mine = [cl for cl in clues if c in clue_cats(cl) and clue_cats(cl) <= set(cats[:i + 1])]
        for perm in itertools.permutations(vals[c]):
            assign[c] = perm
            pos = {(cc, v): assign[cc].index(v) for cc in cats[:i + 1] for v in assign[cc]}
            if all(clue_ok(cl, pos) for cl in mine):
                go(i + 1, assign)
        assign.pop(c, None)
    go(0, {})
    return found


def make_grid(rng, level):
    n, k = {1: (3, 3), 2: (4, 4), 3: (5, 4)}[level]
    cats = ["name", "color", "pet", "drink"][:k]
    vals = {"name": rng.sample(NAMES, n), **{c: rng.sample(VALUES[c], n) for c in cats[1:]}}
    sol = {c: rng.sample(vals[c], n) for c in cats}
    at = lambda c, h: (c, sol[c][h])
    clues = []
    while len(grid_solutions(n, cats, vals, clues)) > 1:
        t = rng.choices(["at", "same", "not", "left", "next", "before"], [1, 3, 2, 2, 2, 1])[0]
        c1, c2 = rng.choice(cats), rng.choice(cats)
        h1 = rng.randrange(n)
        if t == "at":
            clues.append(("at", at(c1, h1), h1))
            continue
        if t == "same":
            if c1 == c2:
                continue
            h2 = h1
        elif t == "not":
            h2 = rng.choice([h for h in range(n) if h != h1])
        elif t == "left":
            if h1 == n - 1:
                continue
            h2 = h1 + 1
        elif t == "next":
            h2 = rng.choice([h for h in (h1 - 1, h1 + 1) if 0 <= h < n])
        else:
            if h1 >= n - 2:
                continue
            h2 = rng.randrange(h1 + 2, n)
        if (c1, h1) != (c2, h2):
            clues.append((t, at(c1, h1), at(c2, h2)))
    for cl in rng.sample(clues, len(clues)):           # drop clues the answer does not need
        rest = [x for x in clues if x != cl]
        if len(grid_solutions(n, cats, vals, rest)) == 1:
            clues = rest
    qc = rng.choice(cats[1:])
    qh = rng.randrange(n)
    ask, ans = at(qc, qh), sol["name"][qh]
    if any(cl[0] == "same" and {cl[1], cl[2]} == {ask, ("name", ans)} for cl in clues):
        return None                                    # the answer is one of the clues: try another seed
    rng.shuffle(clues)
    lines = []
    for t, a, b in clues:
        A = grid_text(*a)
        A = A[0].upper() + A[1:]
        B = f"house {b + 1}" if t == "at" else grid_text(*b)
        lines.append(f"{A} " + {"at": "lives in", "same": "is", "not": "is not", "left": "lives directly to the left of",
                                "next": "lives next to", "before": "lives somewhere to the left of"}[t] + f" {B}.")
    kinds = {"color": "lives in a house of a different color", "pet": "keeps a different pet",
             "drink": "drinks a different drink"}
    intro = (f"There are {n} houses in a row, numbered 1 to {n} from left to right, with one person in each. "
             f"Each person has a different name ({', '.join(sorted(vals['name']))})"
             + "".join(f", {kinds[c]} ({', '.join(sorted(vals[c]))})" for c in cats[1:]) + ".")
    q = {"color": f"Who lives in the {ask[1]} house?", "pet": f"Who owns the {ask[1]}?",
         "drink": f"Who drinks {ask[1]}?"}[qc]
    prompt = intro + "\n\n" + "\n".join(f"{i + 1}. {l}" for i, l in enumerate(lines)) + f"\n\n{q}"
    check = lambda: grid_solutions(n, cats, vals, clues, limit=3) == [{c: tuple(sol[c]) for c in cats}]
    return prompt, "a name", ("text", ans), check


# ---------------------------------------------------------------- scheduling
def sched_best(dur, pre, workers):
    n = len(dur)

    @lru_cache(None)
    def go(done, running):
        """Least time to finish everything from here; running = ((time left, task), ...)."""
        if len(done) == n:
            return 0
        started = done | {t for _, t in running}
        ready = [t for t in range(n) if t not in started and pre[t] <= done]
        free = workers - len(running)
        best = math.inf
        for k in range(min(free, len(ready)), -1, -1):
            for pick in itertools.combinations(ready, k):
                run = list(running) + [(dur[t], t) for t in pick]
                if not run:
                    continue
                step = min(r for r, _ in run)
                fin = frozenset(t for r, t in run if r == step)
                rest = tuple(sorted((r - step, t) for r, t in run if r > step))
                best = min(best, step + go(done | fin, rest))
        return best
    return go(frozenset(), ())


def sched_lower(dur, pre, workers):
    @lru_cache(None)
    def finish(t):
        return dur[t] + max((finish(p) for p in pre[t]), default=0)
    return max(max(finish(t) for t in range(len(dur))), math.ceil(sum(dur) / workers))


def sched_brute(dur, pre, workers):
    """Second method: every order of starting the tasks, each started at its earliest time on the first free worker,
    plus (to allow waiting) every order with one task's start delayed to the next finish.  Small n only."""
    n, best = len(dur), math.inf
    for order in itertools.permutations(range(n)):
        if any(not pre[t] <= set(order[:i]) for i, t in enumerate(order)):
            continue
        free, end = [0] * workers, {}
        for t in order:
            w = min(range(workers), key=lambda i: free[i])
            s = max([free[w]] + [end[p] for p in pre[t]])
            end[t] = s + dur[t]
            free[w] = end[t]
        best = min(best, max(end.values()))
    return best


def make_sched(rng, level):
    n, workers = {1: (6, 6), 2: (6, 2), 3: (8, 3)}[level]
    dur = [rng.randint(1, 9) for _ in range(n)]
    pre = [frozenset(rng.sample(range(t), min(t, rng.choice([0, 1, 1, 2])))) for t in range(n)]
    best = sched_best(dur, pre, workers)
    if workers < n and best == sched_lower(dur, pre, workers):
        return None                                    # the plain lower bound gives it away: try another seed
    letters = rng.sample("ABCDEFGHJK", n)
    lines = []
    for t in rng.sample(range(n), n):
        after = sorted(letters[p] for p in pre[t])
        lines.append(f"Task {letters[t]}: {dur[t]} hour{'s' * (dur[t] > 1)}" + (f", after {' and '.join(after)}" if after else "")
                     + ".")
    who = ("There are enough workers to do any number of tasks at the same time." if workers >= n else
           f"There are {workers} workers. A worker does one task at a time, and each task needs one worker.")
    prompt = (f"A project has {n} tasks. A task can start only when all the tasks it comes after have finished, and "
              f"once started it runs to the end without a break. {who}\n\n" + "\n".join(sorted(lines))
              + "\n\nWhat is the least number of hours, from the start, in which all the tasks can be finished?")
    return prompt, "a whole number of hours", ("int", best), lambda: sched_brute(dur, pre, workers) == best


# ---------------------------------------------------------------- shortest path
SYL = ["dor", "vik", "mal", "len", "ash", "tor", "bri", "nel", "kas", "pel", "run", "ost", "fen", "gal", "hol", "ira"]


def dijkstra(n, edges, s):
    import heapq
    adj = defaultdict(list)
    for (a, b), w in edges.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    d, q = {s: 0}, [(0, s)]
    while q:
        x, u = heapq.heappop(q)
        if x > d[u]:
            continue
        for v, w in adj[u]:
            if x + w < d.get(v, math.inf):
                d[v] = x + w
                heapq.heappush(q, (x + w, v))
    return d


def floyd(n, edges):
    d = [[0 if i == j else math.inf for j in range(n)] for i in range(n)]
    for (a, b), w in edges.items():
        d[a][b] = d[b][a] = min(d[a][b], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def make_path(rng, level):
    n, m = {1: (6, 8), 2: (10, 17), 3: (14, 24)}[level]
    names = set()
    while len(names) < n:
        names.add((rng.choice(SYL) + rng.choice(SYL)).capitalize())
    names = sorted(names)
    rng.shuffle(names)
    edges = {}
    for v in range(1, n):
        edges[tuple(sorted((v, rng.randrange(v))))] = rng.randint(5, 60)
    while len(edges) < m:
        a, b = rng.sample(range(n), 2)
        edges.setdefault(tuple(sorted((a, b))), rng.randint(5, 60))
    fw = floyd(n, edges)
    s, t = max(itertools.combinations(range(n), 2), key=lambda p: (fw[p[0]][p[1]], rng.random()))
    extra, use = "", dict(edges)
    if level == 3:
        d0 = dijkstra(n, edges, s)
        on_path = [e for e, w in edges.items() if abs(d0[e[0]] - d0[e[1]]) == w and
                   min(d0[e[0]], d0[e[1]]) + w + dijkstra(n, edges, t)[e[0] if d0[e[0]] > d0[e[1]] else e[1]] == d0[t]]
        closed = rng.choice(on_path)
        use.pop(closed)
        ds, dt = dijkstra(n, use, s), dijkstra(n, use, t)
        if t not in ds:
            return None
        via = [z for z in range(n) if z not in (s, t) and ds[z] + dt[z] > ds[t]]
        if not via:
            return None
        z = rng.choice(via)
        ans = ds[z] + dt[z]
        extra = (f" Today the road between {names[closed[0]]} and {names[closed[1]]} is closed, and the trip must "
                 f"pass through {names[z]} (any road or town may be used more than once).")
        fw2 = floyd(n, use)
        check = lambda: fw2[s][z] + fw2[z][t] == ans
    else:
        ans = dijkstra(n, edges, s)[t]
        check = lambda: fw[s][t] == ans
    lines = [f"{names[a]} - {names[b]}: {w} km" for (a, b), w in edges.items()]
    rng.shuffle(lines)
    prompt = (f"These are all the roads between {n} towns; each can be driven both ways.\n\n" + "\n".join(lines)
              + f"\n\nWhat is the length of the shortest trip from {names[s]} to {names[t]}?{extra}")
    return prompt, "a whole number of km", ("int", ans), check


# ---------------------------------------------------------------- dates and time zones
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ZONES = {  # name: (tz, standard offset in minutes, rule, text)
    "New York": ("America/New_York", -300, "us", "UTC-5"),
    "Los Angeles": ("America/Los_Angeles", -480, "us", "UTC-8"),
    "London": ("Europe/London", 0, "eu", "UTC+0"),
    "Berlin": ("Europe/Berlin", 60, "eu", "UTC+1"),
    "Sydney": ("Australia/Sydney", 600, "au", "UTC+10"),
    "Tokyo": ("Asia/Tokyo", 540, None, "UTC+9"),
    "Kolkata": ("Asia/Kolkata", 330, None, "UTC+5:30"),
}
RULES = {"us": "one hour more from 2:00 local time on the second Sunday of March until 2:00 local time on the first "
               "Sunday of November",
         "eu": "one hour more from 01:00 UTC on the last Sunday of March until 01:00 UTC on the last Sunday of "
               "October",
         "au": "one hour more from 2:00 local standard time on the first Sunday of October until 3:00 local "
               "daylight time on the first Sunday of April"}


def nth_sunday(y, m, k):
    d = date(y, m, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(weeks=k - 1)


def last_sunday(y, m):
    d = date(y, m + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - 6) % 7)


def rule_offset(city, utc):
    """Second method: the offset (minutes) from the rule text in the prompt, not from the tz database."""
    _, std, rule, _ = ZONES[city]
    u = utc.replace(tzinfo=None)
    y = u.year
    if rule == "us":
        on = datetime.combine(nth_sunday(y, 3, 2), datetime.min.time()) + timedelta(hours=2, minutes=-std)
        off = datetime.combine(nth_sunday(y, 11, 1), datetime.min.time()) + timedelta(hours=2, minutes=-std - 60)
        return std + 60 * (on <= u < off)
    if rule == "eu":
        on = datetime.combine(last_sunday(y, 3), datetime.min.time()) + timedelta(hours=1)
        off = datetime.combine(last_sunday(y, 10), datetime.min.time()) + timedelta(hours=1)
        return std + 60 * (on <= u < off)
    if rule == "au":
        off = datetime.combine(nth_sunday(y, 4, 1), datetime.min.time()) + timedelta(hours=3, minutes=-std - 60)
        on = datetime.combine(nth_sunday(y, 10, 1), datetime.min.time()) + timedelta(hours=2, minutes=-std)
        return std + 60 * (not off <= u < on)
    return std


def zone_text(city):
    _, _, rule, std = ZONES[city]
    return f"{city} is {std}" + (f", {RULES[rule]}" if rule else ", all year") + "."


def make_date(rng, level, inst):
    y0 = date(2027, 1, 1)
    if level == 1 and inst == "a":
        d, k = y0 + timedelta(days=rng.randrange(2900)), rng.randint(400, 2500)
        e = d + timedelta(days=k)
        prompt = f"Which date is {k} days after {WD[d.weekday()]}, {d.isoformat()}, and what day of the week is it?"
        e2 = date.fromordinal(d.toordinal() + k)
        return prompt, "YYYY-MM-DD, Weekday", ("datewd", f"{e.isoformat()} {WD[e.weekday()]}"), \
            lambda: e2 == e and WD[(d.weekday() + k) % 7] == WD[e.weekday()]
    if level == 1:
        d1 = y0 + timedelta(days=rng.randrange(2000))
        d2 = d1 + timedelta(days=rng.randint(500, 2500))
        prompt = f"How many days are there from {d1.isoformat()} to {d2.isoformat()} (the second date minus the first)?"
        ans = (d2 - d1).days
        return prompt, "a whole number", ("int", ans), \
            lambda: sum((date(y, 12, 31) - date(y, 1, 1)).days + 1 for y in range(d1.year, d2.year)) \
            - d1.timetuple().tm_yday + d2.timetuple().tm_yday == ans
    if level == 2:
        d = y0 + timedelta(days=rng.randrange(2900))
        hol = sorted({d + timedelta(days=rng.randint(1, 60)) for _ in range(7)})
        hol = [h for h in hol if h.weekday() < 5][:5]
        work = lambda x: x.weekday() < 5 and x not in hol
        htext = ", ".join(f"{h.isoformat()} ({WD[h.weekday()]})" for h in hol)
        if inst == "a":
            k = rng.randint(15, 35)
            x, c = d, 0
            while c < k:
                x += timedelta(days=1)
                c += work(x)
            prompt = (f"Working days are Monday to Friday, except these holidays: {htext}. Today is "
                      f"{WD[d.weekday()]}, {d.isoformat()}. A job takes {k} working days, starting tomorrow (or the "
                      f"next working day). On which date is its last working day?")
            days = [d + timedelta(days=i) for i in range(1, 120)]
            return prompt, "YYYY-MM-DD", ("date", x.isoformat()), \
                lambda: [y for y in days if work(y)][k - 1] == x
        e = d + timedelta(days=rng.randint(30, 70))
        ans = sum(work(d + timedelta(days=i)) for i in range((e - d).days + 1))
        prompt = (f"Working days are Monday to Friday, except these holidays: {htext}. How many working days are "
                  f"there from {WD[d.weekday()]}, {d.isoformat()} to {WD[e.weekday()]}, {e.isoformat()}, counting "
                  f"both dates?")
        full_weeks = ((e - d).days + 1) // 7
        rest = [d + timedelta(days=7 * full_weeks + i) for i in range((e - d).days + 1 - 7 * full_weeks)]
        return prompt, "a whole number", ("int", ans), \
            lambda: 5 * full_weeks + sum(x.weekday() < 5 for x in rest) - sum(d <= h <= e for h in hol) == ans
    a, b = rng.sample(sorted(ZONES), 2)
    if not (ZONES[a][2] or ZONES[b][2]):
        return None
    rule = ZONES[a][2] or ZONES[b][2]
    year = rng.choice([2027, 2028])
    change = {"us": [nth_sunday(year, 3, 2), nth_sunday(year, 11, 1)], "eu": [last_sunday(year, 3),
              last_sunday(year, 10)], "au": [nth_sunday(year, 4, 1), nth_sunday(year, 10, 1)]}[rule]
    dep_day = rng.choice(change) + timedelta(days=rng.randint(-2, 1))
    dep = datetime.combine(dep_day, datetime.min.time()) + timedelta(hours=rng.randint(6, 22),
                                                                   minutes=rng.choice(range(0, 60, 5)))
    fly = timedelta(hours=rng.randint(5, 16), minutes=rng.choice(range(0, 60, 5)))
    za, zb = ZoneInfo(ZONES[a][0]), ZoneInfo(ZONES[b][0])
    arr = (dep.replace(tzinfo=za).astimezone(timezone.utc) + fly).astimezone(zb).replace(tzinfo=None)
    if dep.replace(tzinfo=za).replace(fold=1).utcoffset() != dep.replace(tzinfo=za).utcoffset():
        return None                                    # departure in an ambiguous hour
    h, mnt = divmod(int(fly.total_seconds()) // 60, 60)
    prompt = (f"{zone_text(a)} {zone_text(b)}\n\nA flight leaves {a} at {dep.strftime('%H:%M')} local time on "
              f"{WD[dep.weekday()]}, {dep.date().isoformat()}, and flies for {h} hours {mnt} minutes. What is the "
              f"local date and time in {b} when it lands?")

    def check():
        u = dep - timedelta(minutes=ZONES[a][1])
        u = dep - timedelta(minutes=rule_offset(a, u))        # a second pass settles a departure near a change
        u = dep - timedelta(minutes=rule_offset(a, u))
        return (u + fly) + timedelta(minutes=rule_offset(b, u + fly)) == arr
    return prompt, "YYYY-MM-DD HH:MM", ("datetime", arr.strftime("%Y-%m-%d %H:%M")), check


# ---------------------------------------------------------------- word problems
def make_word(rng, level, inst):
    F = Fraction
    if level == 1 and inst == "a":
        a, b, d = rng.randint(6, 15), rng.randint(12, 30), rng.choice([10, 20, 25, 40, 60, 75])
        c = rng.randint(a * b // 4, a * b * 3 // 4)
        if (a * b - c) * d % 100:
            return None
        rest = a * b - c
        ans = rest - rest * d // 100
        prompt = (f"A bakery bakes {a} trays of {b} rolls each morning. It sells {c} rolls before noon and {d}% of "
                  f"the rolls that are left during the afternoon. How many rolls are left at closing time?")
        return prompt, "a whole number", ("int", ans), lambda: F(rest) * (1 - F(d, 100)) == ans
    if level == 1:
        p, x, y = rng.randint(300, 900), rng.randint(15, 40), rng.randint(45, 90)
        day, left = 0, p
        while left > 0:
            left -= y if day % 7 in (5, 6) else x
            day += 1
        prompt = (f"A book has {p} pages. Sam starts it on a Monday and reads {x} pages each weekday (Monday to "
                  f"Friday) and {y} pages each day of the weekend. On which day of reading does Sam finish the book "
                  f"(the first Monday is day 1)?")
        week = 5 * x + 2 * y
        full = (p - 1) // week
        r, k = p - full * week, 0
        while r > 0:
            r -= y if k in (5, 6) else x
            k += 1
        return prompt, "a whole number", ("int", day), lambda: full * 7 + k == day
    if level == 2 and inst == "a":
        a, b, c = rng.randint(20, 60), rng.randint(20, 60), rng.randint(30, 90)
        t1, t2 = rng.randint(3, 10), rng.randint(12, 20)
        ra, rb, rc = F(1, a), F(1, b), F(1, c)
        level_t2 = ra * t2 + rb * (t2 - t1)
        if ra * t1 >= 1 or level_t2 >= 1 or ra + rb - rc <= 0:
            return None
        ans = t2 + (1 - level_t2) / (ra + rb - rc)
        prompt = (f"Pipe A fills an empty tank in {a} minutes, pipe B fills it in {b} minutes, and drain C empties a "
                  f"full tank in {c} minutes. The tank starts empty. Pipe A opens at the start, pipe B opens {t1} "
                  f"minutes after the start, and drain C opens {t2} minutes after the start. All stay open. How many "
                  f"minutes after the start is the tank full?")
        sim = lambda: (lambda T: ra * T + rb * (T - t1) - rc * (T - t2))(ans) == 1
        return prompt, "the exact number of minutes (a fraction is fine)", ("frac", ans), sim
    if level == 2:
        v, p = rng.randint(4, 12), rng.choice([5, 8, 10, 12, 15, 20])
        x, y, q = rng.randint(1, v - 2), rng.randint(1, 4), rng.choice([30, 40, 50, 60])
        sugar = F(v * p, 100) - F(x * p, 100) + F(y * q, 100)
        ans = sugar / (v + y) * 100
        prompt = (f"A jug holds {v} liters of juice that is {p}% sugar. You pour out {x} liters, top the jug back up "
                  f"with {x} liters of water, then add {y} liters of syrup that is {q}% sugar. What percent of the "
                  f"mix is sugar now?")
        return prompt, "the exact percent (a fraction is fine)", ("frac", ans), \
            lambda: F(p * (v - x) + q * y, v + y) == ans
    if inst == "a":
        n, c, m = rng.randint(80, 300), rng.choice(range(4, 60, 4)), rng.choice(range(20, 90, 5))
        s, d, k = rng.randint(n // 4, n // 2), rng.choice(range(10, 50, 5)), rng.randint(3, 15)
        r, fee = rng.choice([40, 50, 60, 70]), rng.randint(50, 400)
        price = c * F(100 + m, 100)
        sale = price * F(100 - d, 100)
        ans = s * price + (n - s - k) * sale + k * c * F(r, 100) - n * c - fee
        prompt = (f"A shop buys {n} lamps at ${c} each and pays a ${fee} delivery fee. It marks each lamp up {m}% "
                  f"above cost and sells {s} lamps at that price. It then cuts the price of the rest by {d}% and "
                  f"sells all of them except {k}, which the supplier takes back, refunding {r}% of their cost. What "
                  f"is the shop's profit in dollars (negative for a loss)?")
        alt = lambda: (s * c * (100 + m) * 100 + (n - s - k) * c * (100 + m) * (100 - d) + k * c * r * 100
                       - n * c * 10000 - fee * 10000) == ans * 10000
        return prompt, "the exact number of dollars", ("frac", ans), alt
    D, v1, v2 = rng.randint(60, 140), rng.randint(15, 30), rng.randint(12, 28)
    t0, late, x, brk = rng.randint(7 * 60, 10 * 60), rng.randint(10, 40), rng.randint(15, 40), rng.randint(10, 30)
    # A: rides x km, rests brk minutes, rides on.  B starts `late` minutes after A.  Times in minutes from A's start.
    events = sorted({late, F(x * 60, v1), F(x * 60, v1) + brk})

    def gap(T):
        ta = min(T, F(x * 60, v1)) + max(0, T - F(x * 60, v1) - brk)
        return D - F(v1, 60) * ta - F(v2, 60) * max(0, T - late)
    T0 = 0
    for e in events + [10 ** 6]:
        if gap(e) <= 0:
            lo, hi = T0, e                     # gap is linear on [lo, hi]
            ans = lo + (hi - lo) * gap(lo) / (gap(lo) - gap(hi))
            break
        T0 = e
    if ans <= F(x * 60, v1) + brk or (ans * 60).denominator != 1:
        return None                            # meet after the break, at a whole second
    clock = t0 * 60 + ans * 60
    hh, mm, ss = int(clock // 3600), int(clock % 3600 // 60), int(clock % 60)
    prompt = (f"Two cyclists ride toward each other on the same road between towns {D} km apart. Ana leaves at "
              f"{t0 // 60:02d}:{t0 % 60:02d} riding at {v1} km/h; after {x} km she stops for a {brk}-minute break, "
              f"then rides on at the same speed. Ben leaves the other town {late} minutes after Ana, riding at {v2} "
              f"km/h without stopping. At what time do they meet?")
    alt = lambda: abs(float(F(v1, 60) * (ans - brk) + F(v2, 60) * (ans - late) - D)) < 1e-12
    return prompt, "HH:MM:SS", ("clock", f"{hh:02d}:{mm:02d}:{ss:02d}"), alt


# ---------------------------------------------------------------- unit conversions
INCH = Fraction(254, 10000)                 # m
FOOT, YARD, MILE = 12 * INCH, 36 * INCH, 63360 * INCH
GALLON = 231 * INCH ** 3 * 1000             # liters
LB, G = Fraction(45359237, 10 ** 8), Fraction(980665, 10 ** 5)
DEFS = ("Use exactly: 1 inch = 2.54 cm, 1 foot = 12 inches, 1 yard = 3 feet, 1 mile = 5280 feet, 1 US gallon = 231 "
        "cubic inches, 1 liter = 1000 cubic cm, 1 pound (lb) = 0.45359237 kg, and g = 9.80665 m/s^2.")


def rounded(x, k):
    """x rounded half up to k decimals, as a string, or None if x is within 2% of a step of a rounding boundary."""
    s = x * 10 ** k
    f = s - math.floor(s)
    if abs(f - Fraction(1, 2)) < Fraction(1, 50):
        return None
    return str((Decimal(s.numerator) / Decimal(s.denominator)).quantize(Decimal(1), ROUND_HALF_UP).scaleb(-k))


def make_unit(rng, level, inst):
    F = Fraction
    if level == 1 and inst == "a":
        x = F(rng.randint(180, 450), 10)
        val, k = x * MILE / 1000 / GALLON, 2
        q = f"A car goes {float(x)} miles per US gallon. How many kilometers per liter is that?"
        alt = x * F(1609344, 10 ** 6) / F(3785411784, 10 ** 9)
    elif level == 1:
        ft, inch = rng.randint(4, 6), F(rng.randint(0, 23), 2)
        val, k = (ft * 12 + inch) * F(254, 100), 1
        q = f"How tall in centimeters is a person who is {ft} feet {float(inch)} inches tall?"
        alt = ft * F(3048, 100) + inch * F(254, 100)
    elif level == 2 and inst == "a":
        x = F(rng.randint(180, 450), 10)
        val, k = 100 / (x * MILE / 1000 / GALLON), 2
        q = f"A car goes {float(x)} miles per US gallon. What is its fuel use in liters per 100 km?"
        alt = F(100) * F(3785411784, 10 ** 9) / (x * F(1609344, 10 ** 6))
    elif level == 2:
        x = F(rng.randint(150, 1200), 10)
        val, k = x * LB * G / INCH ** 2 / 1000, 1
        q = (f"A tire is at {float(x)} pounds per square inch (pound-force, the weight of 1 lb under g). What is "
             f"that pressure in kilopascals?")
        alt = x * F(6894757293168361, 10 ** 15)
        if abs(val - alt) > F(1, 10 ** 12):
            return None
        alt = val
    elif inst == "a":
        gpm, head = rng.randint(20, 400), rng.randint(15, 300)
        val, k = gpm * GALLON / 60 * G * head * FOOT, 1
        q = (f"A pump lifts {gpm} US gallons of water per minute through a height of {head} feet. Water has a "
             f"density of exactly 1 kg per liter. How many watts does the lifting take, not counting losses?")
        alt = F(gpm) * F(3785411784, 10 ** 9) / 60 * G * head * F(3048, 10 ** 4)
    else:
        a, v1 = rng.randint(3, 15), rng.randint(8, 20)
        b, t = rng.randint(5, 25), rng.randint(15, 70)
        r, c = rng.randint(5, 20), rng.randint(500, 3000)
        v3 = rng.randint(3, 6)
        dist = a * MILE + b * 1000 + c * YARD
        time = F(a, v1) * 3600 + t * 60 + r * 60 + c * YARD / (v3 * 1000) * 3600
        val, k = dist / time, 3
        q = (f"A rider covers {a} miles at {v1} miles per hour, then {b} km in {t} minutes, stops for {r} minutes, "
             f"then walks the bike {c} yards at {v3} km/h. What is the average speed for the whole trip, stops "
             f"included, in meters per second?")
        alt = (a * F(1609344, 1000) + b * 1000 + c * F(9144, 10 ** 4)) / \
              (F(a * 3600, v1) + (t + r) * 60 + c * F(9144, 10 ** 4) * F(3600, v3 * 1000))
    ans = rounded(val, k)
    if ans is None:
        return None
    prompt = f"{DEFS}\n\n{q} Round to {k} decimal place{'s' if k > 1 else ''}."
    return prompt, f"a number with {k} decimal place{'s' if k > 1 else ''}", ("dec", ans), lambda: alt == val


# ---------------------------------------------------------------- the set
FAMILIES = {"grid": lambda r, l, i: make_grid(r, l), "sched": lambda r, l, i: make_sched(r, l),
            "path": lambda r, l, i: make_path(r, l), "date": make_date, "word": make_word, "unit": make_unit}


def puzzles():
    out = []
    for fam, make in FAMILIES.items():
        for level in (1, 2, 3):
            for inst in "ab":
                for tries in itertools.count():
                    rng = random.Random(f"p22-think-{fam}{level}{inst}-{tries}")
                    p = make(rng, level, inst)
                    if p:
                        break
                prompt, fmt, (kind, ans), check = p
                out.append({"puzzle": f"{fam}{level}{inst}", "family": fam, "level": level, "prompt":
                            prompt + TAIL.format(fmt=fmt), "kind": kind, "answer": str(ans), "check": check})
    return out


# ---------------------------------------------------------------- grading
NUM = r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?"


def final_answer(content):
    hits = re.findall(r"ANSWER\s*:\s*(.+)", content, re.I)
    return hits[-1].replace("*", "").replace("`", "").replace("$", "").strip() if hits else None


def to_frac(s):
    s = s.replace(",", "").replace(" ", "")
    if "/" in s:
        a, b = s.split("/")
        return Fraction(Fraction(a), Fraction(b))
    return Fraction(s)


def right(kind, want, got):
    if got is None:
        return False
    if kind == "code":                                   # img.py: a code or a field's text, spaces and case ignored
        norm = lambda x: re.sub(r"\s", "", x).strip(".\"'").upper()
        return norm(got) == norm(want) or any(t.upper() == want.upper() for t in re.findall(r"[\w./:-]+", got))
    if kind == "set":                                    # img.py: labels in any order, "|" between them in want
        items = {x.strip(" .\"'").casefold() for x in re.split(r",|;|\band\b", got) if x.strip(" .\"'")}
        return items == {w.casefold() for w in want.split("|")}
    if kind == "pair":                                   # img.py: "color shape", nothing else of either kind
        import img
        words = re.findall(r"[a-z]+", got.casefold())
        return ({w for w in words if w in img.COLORS} == {want.split()[0]} and
                {w.rstrip("s") for w in words if w.rstrip("s") in img.SHAPES} == {want.split()[1]})
    if kind == "rc":                                     # img.py: row, column
        return re.findall(r"\d+", got)[:2] == want.split(",")
    if kind == "text":
        return got.strip(" .").casefold() == want.casefold()
    if kind in ("date", "datetime", "clock", "datewd"):
        pat = {"date": r"\d{4}-\d{2}-\d{2}", "datetime": r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}",
               "clock": r"\d{1,2}:\d{2}:\d{2}", "datewd": r"\d{4}-\d{2}-\d{2}"}[kind]
        m = re.search(pat, got)
        if not m:
            return False
        v = m.group(0).replace("T", " ")
        if kind == "datetime":
            v = v[:11] + v[11:].zfill(5)
        if kind == "clock":
            v = v.zfill(8)
        if kind == "datewd":
            wd = re.findall(r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday", got, re.I)
            return v == want[:10] and len(wd) == 1 and wd[0].casefold() == want[11:].casefold()
        return v == want
    m = re.search(NUM, got)
    if not m:
        return False
    try:
        x = to_frac(m.group(0))
    except (ValueError, ZeroDivisionError):
        return False
    if kind == "int":
        return x == int(want)
    if kind == "frac":
        return x == to_frac(want)
    k = len(want.split(".")[1]) if "." in want else 0
    return (Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal(1).scaleb(-k), ROUND_HALF_UP) \
        == Decimal(want)


def selftest():
    bad, ps = 0, puzzles()
    for p in ps:
        ok = p["check"]()
        good = f"blah\nANSWER: {p['answer'] if p['kind'] != 'datewd' else p['answer'].replace(' ', ', ')}"
        g1, g0 = right(p["kind"], p["answer"], final_answer(good)), right(p["kind"], p["answer"], final_answer("no"))
        print(f"{p['puzzle']:8} {'ok ' if ok and g1 and not g0 else 'BAD'} {p['kind']:8} {p['answer']}")
        bad += not (ok and g1 and not g0)
    for kind, want, got, exp in (("frac", "125/3", "41 2/3", False), ("frac", "125/3", "125/3 minutes", True),
                                 ("frac", "-1234.5", "-$1,234.50", True), ("dec", "12.35", "12.35 km/L", True),
                                 ("dec", "12.35", "12.3", False), ("datetime", "2027-03-14 07:05", "2027-03-14 7:05",
                                                                    True), ("text", "Ava", "Ava.", True),
                                 ("int", "41", "about 41 hours", True),
                                 ("code", "KX4R9T", "kx4r 9t", True), ("code", "KX4R9T", "KX4R9", False),
                                 ("code", "nova-27.lan", "The host is nova-27.lan", True),
                                 ("set", "Sound|Backups", "Backups, Sound", True), ("set", "Sound|Backups", "Sound",
                                                                                    False),
                                 ("pair", "red circle", "a red circle", True), ("pair", "red circle", "red square",
                                                                                False),
                                 ("pair", "red circle", "red or blue circle", False), ("rc", "2,3", "row 2, column 3",
                                                                                       True), ("datewd", "2030-01-01 Tuesday",
                                                                         "2030-01-01, Tuesday", True)):
        if right(kind, want, got.replace("$", "")) != exp:
            print("grader BAD", kind, want, got)
            bad += 1
    print(f"{len(ps)} puzzles, {bad} bad")
    return bad


def main():
    cmd = sys.argv[1]
    if cmd == "selftest":
        sys.exit(1 if selftest() else 0)
    if cmd == "prompts":
        out = Path(sys.argv[2])
        for p in puzzles():
            for mode, req in MODES.items():
                d = out / f"{p['puzzle']}-{mode}"
                d.mkdir(parents=True, exist_ok=True)
                t = {k: v for k, v in p.items() if k != "check"}
                (d / "task.json").write_text(json.dumps({**t, "id": d.name, "kind": "think", "mode": mode,
                                                         "answer_kind": p["kind"], "request": req}, indent=1))
        return
    if cmd == "grade":
        tasks = Path(sys.argv[2])
        for runs in sys.argv[3:]:
            out = []
            for line in Path(runs).read_text().splitlines():
                r = json.loads(line)
                t = json.loads((tasks / r["id"] / "task.json").read_text())
                got = final_answer(r["content"])
                ok = right(t["answer_kind"], t["answer"], got)
                out.append({k: v for k, v in r.items() if k not in ("content", "reasoning")} |
                           {"puzzle": t["puzzle"], "family": t["family"], "level": t["level"], "mode": t["mode"],
                            "want": t["answer"], "got": got, "pass": ok, "think_chars": len(r["reasoning"])})
            Path(runs).with_suffix(".graded.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
        return
    if cmd == "score":
        runs = [[json.loads(l) for l in Path(p).read_text().splitlines()] for p in sys.argv[2:]]
        names = [rows[0]["model"] for rows in runs]
        every = [r for rows in runs for r in rows]
        modes = list(dict.fromkeys(r["mode"] for r in every))
        print("| | " + " | ".join(f"{n} {m}" for n in names for m in modes) + " |")
        print("|---|" + "---|" * (len(names) * len(modes)))
        groups = [(f, lambda r, f=f: r["family"] == f) for f in dict.fromkeys(r["family"] for r in every)] + \
                 [(f"level {l}", lambda r, l=l: r["level"] == l) for l in sorted({r["level"] for r in every})] + \
                 [("all", lambda r: True)]
        for label, pick in groups:
            cells = []
            for rows in runs:
                for m in modes:
                    mine = [r for r in rows if r["mode"] == m and pick(r)]
                    n = len({(r["puzzle"]) for r in mine}) or 1
                    cells.append(f"{sum(r['pass'] for r in mine) / max(1, len(mine)) * n:.1f} of {n}"
                                 if mine else "-")
            print(f"| {label} | " + " | ".join(cells) + " |")
        for label, fn in (("output tokens, median", lambda xs: sorted(r.get("completion_tokens") or 0 for r in xs)
                           [len(xs) // 2]),
                          ("time per answer, median s", lambda xs: sorted(r.get("wall_s") or 0 for r in xs)
                           [len(xs) // 2]),
                          ("time for all, min", lambda xs: round(sum(r.get("wall_s") or 0 for r in xs) / 60)),
                          ("cut off at the limit", lambda xs: sum(r.get("finish") == "length" for r in xs)),
                          ("no ANSWER line", lambda xs: sum(r["got"] is None for r in xs))):
            cells = [str(fn([r for r in rows if r["mode"] == m]) if any(r["mode"] == m for r in rows) else "-")
                     for rows in runs for m in modes]
            print(f"| {label} | " + " | ".join(cells) + " |")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
