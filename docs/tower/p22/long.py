"""Pass 22, long context and stability: questions about notes hidden in long code, and JSON from short texts.

Long context: PolyStrata's own Python (terminal/serve and terminal/strata_control at f6bc1acf1, tests left out) is
cut to a size, and notes are put in it as comments at chosen depths: which depot a unit is kept at, which town a
depot is in, and changes to a queue's retry limit.  Six questions follow the code, each asked on its own:
  q1, q2  one note, early (10-30% in) and late (70-95% in)
  q3      two notes: the town of the depot a unit is kept at
  q4      a count: how many units are kept at one depot (3 to 6 notes, spread out)
  q5      the last of 4 to 6 changes to a setting, with another queue's changes in between
  q6      the value of a real constant in the code (one assignment of that name in the whole text)
Levels 1-3 are about 50K, 100K and 150K characters (both models); level 4 is about 400K (Cyber-Tiel only:
Inkling's config stops at 65,536 tokens, so it is turned away and not scored).
JSON: 10 short texts about a person, answered as a JSON object of 5 fields, once with only the instruction (`free`)
and once with a `response_format` JSON schema (`schema`, the server's grammar).
    python3 long.py selftest         every answer found again by a second method
    python3 long.py prompts OUT      OUT/<id>/task.json (needs git: run it in the repository)
    python3 think.py grade OUT RUNS... / think.py score GRADED...
    python3 long.py loops RUNS...    per answers file: replies, loops (is_loop: a line repeated 4 times in a row, or an
                                     ending repeated 4 times), cut off at the limit, failed requests
"""
import ast, json, random, re, subprocess, sys
from pathlib import Path

COMMIT = "f6bc1acf1"
LEVELS = {1: 50_000, 2: 100_000, 3: 150_000, 4: 400_000}
MODE = {"reasoning_budget_tokens": 2048, "max_tokens": 4096}
BIRDS = ["KESTREL", "HERON", "PLOVER", "OSPREY", "MERLIN", "SWIFT", "WREN", "EGRET", "CURLEW", "LINNET"]
DEPOTS = ["Varn", "Holt", "Brask", "Teller", "Quorn", "Mirel"]
TOWNS = ["Ostmark", "Fennick", "Dalby", "Corran", "Pellow", "Ruskin"]
QUEUES = ["ingest-b", "export-k"]
TAIL = "\n\nEnd your reply with one last line of the form\nANSWER: {fmt}"
HERE = Path(__file__).parent


def git(*a):
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
    return subprocess.check_output(["git", *a], cwd=root, text=True)


def sources():
    names = [p for p in git("ls-tree", "-r", "--name-only", COMMIT, "terminal/serve", "terminal/strata_control")
             .split() if p.endswith(".py") and "/test_" not in p and "/tests/" not in p]
    return {p: git("show", f"{COMMIT}:{p}") for p in sorted(names)}


def build_level(level, files, tries=0):
    rng = random.Random(f"p22-long-{level}-{tries}")
    budget = LEVELS[level]
    order = sorted(files)
    rng.shuffle(order)
    parts, size = [], 0
    for p in order:
        text = files[p]
        if size + len(text) > budget * 1.02 or len(text) < 1500:
            continue
        parts.append(f"# ===== file: {p} =====\n{text}")
        size += len(text)
        if size > budget * 0.97:
            break
    lines = "\n".join(parts).splitlines()
    blob = "\n".join(lines).casefold()
    assert not any(w.casefold() in blob for w in BIRDS + DEPOTS + TOWNS + QUEUES), "a planted name is in the code"
    units = rng.sample([f"{b}-{n}" for b in BIRDS for n in range(10, 100)], 18)
    target = rng.choice(DEPOTS)
    k = rng.randint(3, 6)
    depot_of = {u: target for u in units[:k]}
    for u in units[k:]:
        depot_of[u] = rng.choice([d for d in DEPOTS if d != target])
    town_of = dict(zip(DEPOTS, rng.sample(TOWNS, len(DEPOTS))))
    notes = []                                         # (depth 0-1, text)
    early, late, hop = units[k], units[k + 1], units[k + 2]
    for u in units:
        depth = rng.uniform(0.1, 0.3) if u == early else rng.uniform(0.7, 0.95) if u == late else rng.uniform(0.02, 0.98)
        notes.append((depth, f"# fleet note: unit {u} is kept at depot {depot_of[u]}"))
    for d in DEPOTS:
        notes.append((rng.uniform(0.02, 0.98), f"# fleet note: depot {d} is in the town of {town_of[d]}"))
    changes = {}
    for q in QUEUES:
        vals = rng.sample(range(2, 40), rng.randint(4, 6))
        depths = sorted(rng.uniform(0.02, 0.98) for _ in vals)
        changes[q] = vals
        for dpt, v in zip(depths, vals):
            notes.append((dpt, f"# ops change: retry limit for queue {q} set to {v}"))
    for depth, text in sorted(notes, reverse=True):    # deepest first, so earlier indices stay right
        lines.insert(int(depth * len(lines)), text)
    text = "\n".join(lines)
    consts = []
    for m in re.finditer(r"^([A-Z][A-Z0-9_]{4,}) = (\d+)\s*(#.*)?$", text, re.M):
        if len(re.findall(rf"^\s*{m.group(1)}\s*[:=]", text, re.M)) == 1 and int(m.group(2)) > 9:
            consts.append((m.group(1), int(m.group(2))))
    if not consts:                                     # no usable constant in these files: another pick
        return build_level(level, files, tries + 1)
    name, value = rng.choice(consts)
    q = rng.choice(QUEUES)
    qs = [("q1", f"At which depot is unit {early} kept?", "the depot's name", ("text", depot_of[early])),
          ("q2", f"At which depot is unit {late} kept?", "the depot's name", ("text", depot_of[late])),
          ("q3", f"In which town is the depot that unit {hop} is kept at?", "the town's name",
           ("text", town_of[depot_of[hop]])),
          ("q4", f"How many units are kept at depot {target}?", "a whole number", ("int", str(k))),
          ("q5", f"The ops changes are listed in the order they were made. What is the retry limit for queue {q} "
                 f"after the last change?", "a whole number", ("int", str(changes[q][-1]))),
          ("q6", f"In the code, what whole number is the module-level constant {name} set to?", "a whole number",
           ("int", str(value)))]
    return text, qs, {"early": early, "late": late, "hop": hop, "target": target, "queue": q, "const": name}


def check_level(text, qs, keys):
    """Second method: read the answers back out of the finished text, and the constant with ast, file by file."""
    kept = dict(re.findall(r"^# fleet note: unit (\S+) is kept at depot (\w+)$", text, re.M))
    town = dict(re.findall(r"^# fleet note: depot (\w+) is in the town of (\w+)$", text, re.M))
    last = {}
    for q, v in re.findall(r"^# ops change: retry limit for queue (\S+) set to (\d+)$", text, re.M):
        last[q] = v
    pos = lambda u: text.index(f"unit {u} is kept") / len(text)
    found = None
    for chunk in text.split("# ===== file: ")[1:]:
        body = re.sub(r"^# (fleet note|ops change):.*$", "", chunk.split("\n", 1)[1], flags=re.M)
        for node in ast.parse(body).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == keys["const"]
                                                    for t in node.targets):
                found = ast.literal_eval(node.value)
    got = [kept[keys["early"]], kept[keys["late"]], town[kept[keys["hop"]]],
           str(sum(d == keys["target"] for d in kept.values())), last[keys["queue"]], str(found)]
    return got == [a for _, _, _, (_, a) in qs] and 0.1 <= pos(keys["early"]) <= 0.31 and pos(keys["late"]) >= 0.69


# ---------------------------------------------------------------- JSON
FIRST = ["Rosa", "Ivan", "Mei", "Tomas", "Aiyana", "Bruno", "Leila", "Kofi", "Sanne", "Diego"]
LAST = ["Lind", "Petrov", "Okafor", "Varga", "Haldane", "Moreau", "Sato", "Quist", "Brandt", "Ferreira"]
HOBBIES = ["cycling", "pottery", "chess", "birdwatching", "rowing", "baking", "climbing", "choir singing"]
JOBS = ["a nurse", "a welder", "a teacher", "a bus driver", "an accountant"]
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "city": {"type": "string"},
                                           "hobbies": {"type": "array", "items": {"type": "string"}},
                                           "employed": {"type": "boolean"}},
          "required": ["name", "age", "city", "hobbies", "employed"], "additionalProperties": False}


def person(i):
    rng = random.Random(f"p22-json-{i}")
    name = f"{FIRST[i]} {LAST[i]}"
    age, old, new = rng.randint(22, 70), *rng.sample(TOWNS, 2)
    hob = rng.sample(HOBBIES, rng.randint(1, 3))
    job = rng.random() < 0.5
    he = rng.random() < 0.5
    P, p = ("He", "his") if he else ("She", "her")
    work = (f"{P} works as {rng.choice(JOBS)}." if job else
            f"{P} has been out of work since the spring and is looking for a job.")
    hobtext = hob[0] if len(hob) == 1 else ", ".join(hob[:-1]) + " and " + hob[-1]
    text = (f"{name} turned {age} last month. {P} grew up in {old} but moved to {new} two years ago and still lives "
            f"there. {work} In {p} free time, {p} hobbies are {hobtext}. {P} has a brother who lives in {old}.")
    want = {"name": name, "age": age, "city": new, "hobbies": hob, "employed": job}
    prompt = (f"{text}\n\nWrite the facts about {name} as a JSON object with exactly these keys: name (string), age "
              f"(integer), city (string, where the person lives now), hobbies (list of strings, in the order given, "
              f"each as written in the text), employed (true or false). Reply with only the JSON object.")
    return prompt, want


def parse_json(content):
    s = content.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, re.S)
    try:
        return json.loads(m.group(1) if m else s)
    except (ValueError, TypeError):
        return None


def json_right(want, content):
    got = parse_json(content)
    if not isinstance(got, dict) or set(got) != set(want):
        return False
    norm = lambda v: [norm(x) for x in v] if isinstance(v, list) else v.strip().casefold() if isinstance(v, str) else v
    return all(norm(got[k]) == norm(want[k]) and type(got[k]) is type(want[k]) for k in want)


def tasks():
    files = sources()
    out = []
    for level in LEVELS:
        text, qs, keys = build_level(level, files)
        head = ("Below are several Python files from one project, with notes in comments. Read them, then answer the "
                "question after them.\n\n```python\n" + text + "\n```\n\nQuestion: ")
        for qid, q, fmt, (kind, ans) in qs:
            out.append({"id": f"long{level}-{qid}", "puzzle": f"long{level}-{qid}", "family": f"long {level}",
                        "level": level, "mode": "2k", "prompt": head + q + TAIL.format(fmt=fmt), "answer": ans,
                        "answer_kind": kind, "request": MODE, "chars": len(text),
                        "check": (text, qs, keys)})
    for i in range(10):
        prompt, want = person(i)
        for mode in ("free", "schema"):
            req = dict(MODE)
            if mode == "schema":
                req["response_format"] = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA}}
            out.append({"id": f"json{i:02d}-{mode}", "puzzle": f"json{i:02d}", "family": "json", "level": 0,
                        "mode": mode, "prompt": prompt, "answer": json.dumps(want), "answer_kind": "json",
                        "request": req})
    return out


def selftest():
    bad, seen = 0, set()
    for t in tasks():
        if "check" in t and t["family"] not in seen:
            seen.add(t["family"])
            ok = check_level(*t["check"])
            print(f"{t['family']}: {t['chars']} chars, answers {[a for *_, (_, a) in t['check'][1]]} "
                  f"{'ok' if ok else 'BAD'}")
            bad += not ok
        if t["answer_kind"] == "json":
            want = json.loads(t["answer"])
            ok = json_right(want, "```json\n" + json.dumps(want) + "\n```") and json_right(want, json.dumps(want)) \
                and not json_right(want, json.dumps({**want, "age": str(want["age"])})) and not json_right(want, "")
            bad += not ok
    print(f"{len(seen)} levels, 10 JSON texts, {bad} bad")
    return bad


def is_loop(text):
    """A reply stuck repeating: the same line of 20+ characters 4 or more times in a row, or an ending that is one
    piece of 20-750 characters said 4 times over."""
    run, prev = 1, None
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        run = run + 1 if line == prev and len(line) >= 20 else 1
        if run >= 4:
            return True
        prev = line
    tail = text.rstrip()[-3000:]
    return any(len(tail) >= 4 * n and tail[-4 * n:] == tail[-n:] * 4 for n in range(20, 751))


def loops(paths):
    print("| answers | replies | loops | cut off | failed |\n|---|---|---|---|---|")
    for path in paths:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
        looped = []
        for r in rows:
            for part in (r.get("reasoning") or "", r.get("content") or ""):
                if is_loop(part):
                    looped.append(r["id"])
                    break
        cut = sum(r.get("finish") == "length" for r in rows)
        failed = sum(str(r.get("finish")).startswith("error") for r in rows)
        print(f"| {Path(path).name} | {len(rows)} | {len(looped)} {' '.join(looped)} | {cut} | {failed} |")


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["loops"]:
        loops(sys.argv[2:])
    elif cmd == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    elif cmd == ["prompts"]:
        out = Path(sys.argv[2])
        for t in tasks():
            d = out / t["id"]
            d.mkdir(parents=True, exist_ok=True)
            t = {k: v for k, v in t.items() if k != "check"}
            (d / "task.json").write_text(json.dumps({**t, "kind": "long"}, indent=1))
    else:
        sys.exit(__doc__)
