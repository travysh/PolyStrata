"""Grade Pass 22 answers.  Runs inside the PolyStrata image with no network:

    docker run --rm --network none -v /mnt/user/appdata/polystrata/p22:/p22 --entrypoint /opt/strata/terminal/.venv/bin/python \
        polystrata:tower /p22/grade.py check  /p22/tasks            # the real fixes pass, no fix fails; writes check.json
    ... /p22/grade.py grade /p22/tasks /p22/runs/<label>.jsonl     # writes <label>.graded.jsonl

Only the answer (`content`) is graded, never the reasoning.  A repo task passes when its SEARCH/REPLACE blocks apply and
every test of the fix's test modules passes, except those that fail with the real fix too (named in check.json).  A
function task passes when its code passes all its hidden tests (see func_tasks.py)."""
import json, os, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

PY = "/opt/strata/terminal/.venv/bin/python"
RUNNER = r"""
import json, sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[2:])
res = unittest.TestResult()
suite.run(res)
bad = sorted({t.id() if hasattr(t, "id") else str(t) for t, _ in res.failures + res.errors})
json.dump({"run": res.testsRun, "bad": bad, "detail": [tb[-1500:] for _, tb in (res.failures + res.errors)[:3]]},
          open(sys.argv[1], "w"))
"""


# ---------------------------------------------------------------- SEARCH/REPLACE blocks

def parse_blocks(text):
    """[(path, search, replace)] from an answer.  The path is the last non-empty, non-fence line before the block."""
    blocks, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if lines[i].strip() == "<<<<<<< SEARCH":
            path = ""
            for back in range(i - 1, max(i - 4, -1), -1):
                s = lines[back].strip().strip("`*#:").strip()
                if s and not s.startswith("```") and s not in ("python", "py"):
                    path = s.split()[-1].strip("`*")
                    break
            j = i + 1
            while j < len(lines) and lines[j].strip() != "=======":
                j += 1
            k = j + 1
            while k < len(lines) and not lines[k].strip().startswith(">>>>>>> REPLACE"):
                k += 1
            if k >= len(lines):
                break
            blocks.append((path, "\n".join(lines[i + 1:j]), "\n".join(lines[j + 1:k])))
            i = k
        i += 1
    return blocks


def resolve(root, path, sources):
    """The file a block names: its path, or the one source file whose path ends with it."""
    if path in sources:
        return path
    hits = [s for s in sources if s.endswith("/" + path.lstrip("./")) or s.endswith(path)]
    if len(hits) == 1:
        return hits[0]
    if len(sources) == 1 and not path.endswith((".py", ".cpp", ".h")):
        return sources[0]
    return path if (root / path).is_file() else None


def apply_blocks(root, answer, sources):
    """Apply every block; (ok, message).  A SEARCH must match exactly once (trailing spaces of lines ignored)."""
    blocks = parse_blocks(answer)
    if not blocks:
        return False, "no SEARCH/REPLACE block"
    for n, (path, search, replace) in enumerate(blocks, 1):
        rel = resolve(root, path, sources)
        if not rel:
            return False, f"block {n}: unknown file {path!r}"
        f = root / rel
        text = f.read_text()
        if not search.strip():
            return False, f"block {n}: empty SEARCH"
        if text.count(search) == 1:
            f.write_text(text.replace(search, replace, 1))
            continue
        strip = lambda s: "\n".join(l.rstrip() for l in s.split("\n"))
        st, ss = strip(text), strip(search)
        if st.count(ss) == 1:
            f.write_text(st.replace(ss, strip(replace), 1))
            continue
        return False, f"block {n}: SEARCH matches {st.count(ss)} times in {rel}"
    return True, f"{len(blocks)} block(s) applied"


# ---------------------------------------------------------------- repo tasks

def run_repo(taskdir, answer, timeout=900):
    task = json.loads((taskdir / "task.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for tar in ("parent.tar", "tests.tar"):
            with tarfile.open(taskdir / tar) as t:
                t.extractall(root, filter="tar")
        if answer is None:
            ok, msg = True, "no change"
        else:
            ok, msg = apply_blocks(root, answer, task["sources"])
        if not ok:
            return {"applied": False, "msg": msg, "bad": None}
        out = root / "result.json"
        env = dict(os.environ, HOME=tmp, PYTHONDONTWRITEBYTECODE="1", STRATA_NO_NETWORK="1")
        try:
            p = subprocess.run([PY, "-c", RUNNER, str(out), *task["modules"]], cwd=root / "terminal", env=env,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"applied": True, "msg": msg + "; tests timed out", "bad": ["TIMEOUT"]}
        if not out.exists():
            return {"applied": True, "msg": msg + "; runner died: " + p.stderr[-800:], "bad": ["RUNNER"]}
        r = json.loads(out.read_text())
        return {"applied": True, "msg": msg, "bad": r["bad"], "run": r["run"], "detail": r["detail"]}


def check(tasks):
    """Each task: the bug's tests fail without a fix and pass with the real fix."""
    out = {}
    for d in sorted(p for p in tasks.iterdir() if (p / "task.json").exists()):
        task = json.loads((d / "task.json").read_text())
        if task["kind"] != "repo":
            continue
        none = run_repo(d, None)
        ref = run_repo(d, task["reference"])
        fixed = sorted(set(none["bad"] or []) - set(ref["bad"] or []))
        out[task["id"]] = {"valid": bool(ref["applied"] and fixed), "fixed_by_ref": fixed,
                           "bad_with_ref": ref["bad"], "bad_without": none["bad"], "ref_msg": ref["msg"],
                           "run": ref.get("run"), "detail_without": none.get("detail", [])[:1]}
        print(task["id"], "VALID" if out[task["id"]]["valid"] else "INVALID", "run", ref.get("run"),
              "fixed", len(fixed), "bad with ref", ref["bad"], ref["msg"], flush=True)
    (tasks / "check.json").write_text(json.dumps(out, indent=1))


def grade(tasks, runs):
    checked = json.loads((tasks / "check.json").read_text())
    sys.path.insert(0, str(Path(__file__).parent))
    import func_tasks
    out = runs.with_suffix(".graded.jsonl")
    with open(runs) as f, open(out, "w") as w:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "repo":
                g = run_repo(tasks / r["id"], r.get("content") or "")
                allowed = set(checked[r["id"]]["bad_with_ref"] or [])
                g["pass"] = bool(g["applied"] and g["bad"] is not None and set(g["bad"]) <= allowed)
            else:
                g = func_tasks.grade(r["id"], r.get("content") or "")
            g.pop("detail", None)
            r.update(grade=g)
            r.pop("content", None)
            r.pop("reasoning", None)
            w.write(json.dumps(r) + "\n")
            print(r["id"], r.get("sample"), "PASS" if g["pass"] else "fail", g.get("msg", "")[:120], flush=True)


if __name__ == "__main__":
    cmd, tasks = sys.argv[1], Path(sys.argv[2])
    if cmd == "check":
        check(tasks)
    elif cmd == "grade":
        grade(tasks, Path(sys.argv[3]))
