"""Build the repo-fix tasks of Pass 22 from real PolyStrata commits.

For each task in repo_tasks.json (a fix commit that came with tests):
  OUT/<id>/parent.tar   `terminal/` at the commit's parent (the code with the bug)
  OUT/<id>/tests.tar    the commit's test files (hidden from the model)
  OUT/<id>/task.json    the prompt, the test modules, and the real fix as SEARCH/REPLACE blocks (to check the grader)
The prompt shows excerpts of the files the fix changed: a window around the changed lines, placed at random (seeded by
the task id) so the change is not in the middle.  Run on the Mac, in the repository:
    python3 docs/tower/p22/build_repo.py OUT [ids...]
"""
import difflib, json, random, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip())
WINDOW_CHARS = 36000          # about 10K tokens of code in the prompt, split over the changed files

FORMAT = """Answer with the change as one or more SEARCH/REPLACE blocks, each exactly like this:

path/of/the/file.py
<<<<<<< SEARCH
lines copied exactly from the file, enough to be unique
=======
the lines that replace them
>>>>>>> REPLACE

The SEARCH lines must match the file exactly (indentation included) and only once. Change only what the fix needs.
You may add new functions or methods inside a REPLACE. Do not write tests."""


def git(*args, binary=False):
    return subprocess.check_output(["git", *args], cwd=REPO, text=not binary)


def is_test(path):
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or "/tests/" in path or name.endswith("_support.py")


def module_of(path):
    """terminal/serve/test_x.py -> serve.test_x (tests run from terminal/)."""
    return path[len("terminal/"):-3].replace("/", ".")


def hunks(parent, commit, path):
    """The parent's line ranges (1-based, inclusive) the commit changed in `path`."""
    out = []
    for line in git("diff", "-U0", parent, commit, "--", path).splitlines():
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line)
        if m:
            start, n = int(m.group(1)), int(m.group(2) or 1)
            out.append((max(start, 1), max(start + n - 1, start)))
    return out


def window(lines, changed, budget, rng):
    """1-based [a, b] around the changed lines, about `budget` characters, with the change at a random place in it."""
    a, b = changed[0][0], changed[-1][1]
    size = lambda x, y: sum(len(l) + 1 for l in lines[x - 1:y])
    room = max(budget - size(a, b), 0)
    before = int(room * rng.uniform(0.2, 0.8))
    while a > 1 and size(a - 1, a - 1) <= before:
        before -= size(a - 1, a - 1)
        a -= 1
    after = budget - size(a, b)
    while b < len(lines) and size(b + 1, b + 1) <= after:
        after -= size(b + 1, b + 1)
        b += 1
    return a, b


def ref_blocks(parent, commit, sources):
    """The real fix as SEARCH/REPLACE blocks (3 lines of context), to check that the grader passes it."""
    blocks = []
    for path in sources:
        old = git("show", f"{parent}:{path}").splitlines()
        new = git("show", f"{commit}:{path}").splitlines()
        for group in difflib.SequenceMatcher(None, old, new, autojunk=False).get_grouped_opcodes(3):
            i1, i2 = group[0][1], group[-1][2]
            j1, j2 = group[0][3], group[-1][4]
            blocks.append(f"{path}\n<<<<<<< SEARCH\n" + "\n".join(old[i1:i2]) + "\n=======\n"
                          + "\n".join(new[j1:j2]) + "\n>>>>>>> REPLACE")
    return "\n\n".join(blocks)


def build(task, out):
    commit = git("rev-parse", task["commit"]).strip()
    parent = git("rev-parse", commit + "^").strip()
    changed = [p for p in git("show", "--name-only", "--format=", commit).split() if p.startswith("terminal/")]
    tests = [p for p in changed if is_test(p)]
    sources = [p for p in changed if not is_test(p) and p.endswith(".py")]
    sources = task.get("sources", sources)
    modules = task.get("modules") or [module_of(p) for p in tests if p.rsplit("/", 1)[-1].startswith("test_")]
    rng = random.Random(task["id"])
    excerpts = []
    for path in sources:
        lines = git("show", f"{parent}:{path}").splitlines()
        size = lambda x, y: sum(len(l) + 1 for l in lines[x - 1:y])
        groups = []                                   # changes far apart in a big file get windows of their own
        for h in hunks(parent, commit, path):
            if groups and size(groups[-1][0][0], h[1]) <= WINDOW_CHARS // (2 * len(sources)):
                groups[-1].append(h)
            else:
                groups.append([h])
        budget = WINDOW_CHARS // (len(sources) * len(groups))
        for g in groups:
            a, b = window(lines, g, budget, rng)
            excerpts.append(f"### {path} (lines {a}-{b} of {len(lines)})\n```python\n" + "\n".join(lines[a - 1:b]) + "\n```")
    prompt = (f"You are fixing a bug in PolyStrata, a Python and C++ server that runs large language models.\n\n"
              f"## Bug report\n\n{task['issue'].strip()}\n\n## Code\n\nExcerpts of the files involved (the files are longer; "
              f"code outside the excerpts exists and works):\n\n" + "\n\n".join(excerpts) + f"\n\n## Your answer\n\n{FORMAT}")
    d = out / task["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "parent.tar").write_bytes(git("archive", parent, "terminal", binary=True))
    (d / "tests.tar").write_bytes(git("archive", commit, *tests, binary=True))
    (d / "task.json").write_text(json.dumps({
        "id": task["id"], "kind": "repo", "commit": commit, "parent": parent, "sources": sources, "tests": tests,
        "modules": modules, "prompt": prompt, "reference": ref_blocks(parent, commit, sources)}, indent=1))
    return d, len(prompt)


def main():
    out = Path(sys.argv[1])
    want = set(sys.argv[2:])
    for task in json.loads((HERE / "repo_tasks.json").read_text()):
        if want and task["id"] not in want:
            continue
        d, n = build(task, out)
        print(f"{task['id']}: {n} prompt chars -> {d}")


if __name__ == "__main__":
    main()
