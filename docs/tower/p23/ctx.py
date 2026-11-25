"""Pass 23, how long a context Inkling 2 Mini can use on Tower: Pass 22's long-context questions (p22/long.py: notes
hidden in PolyStrata's Python, six questions a level) at sizes past 64K tokens.
Levels (characters of code; Inkling makes about 3.06 characters a token of this code):
  5   400,000  ~130K tokens      6   750,000  ~245K tokens
  7 1,500,000  ~490K tokens      8 3,000,000  ~980K tokens
    python3 ctx.py selftest        every answer found again by a second method
    python3 ctx.py prompts OUT     OUT/<id>/task.json (run inside the repository)
    python3 ctx.py swap OUT        two ~128K-token texts asked in turn, A B A A B (a parked conversation brought back
                                   after the other one was read: OUT/s1-A-q1 ... s5-B-q2)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "p22"))
import long

long.LEVELS = {5: 400_000, 6: 750_000, 7: 1_500_000, 8: 3_000_000}
_src = long.sources
WORDS = [w.casefold() for w in long.BIRDS + long.DEPOTS + long.TOWNS + long.QUEUES]
# the larger levels reach files that hold a planted word (Swift, Wren, ...): those files are left out
long.sources = lambda: {p: t for p, t in _src().items() if not any(w in t.casefold() for w in WORDS)}

if __name__ == "__main__":
    tasks = [t for t in long.tasks() if t["family"] != "json"]
    if sys.argv[1:2] == ["selftest"]:
        for t in tasks:
            if t["id"].endswith("q1"):
                print(t["family"], t["chars"], "ok" if long.check_level(*t["check"]) else "BAD")
    elif sys.argv[1:2] == ["prompts"]:
        for t in tasks:
            d = Path(sys.argv[2]) / t["id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "task.json").write_text(json.dumps({**{k: v for k, v in t.items() if k != "check"}, "kind": "long"}))
    elif sys.argv[1:2] == ["swap"]:
        files = long.sources()
        head = "Below are several Python files from one project, with notes in comments. Read them, then answer the " \
               "question after them.\n\n```python\n"
        docs = {"A": long.build_level(5, files), "B": long.build_level(5, files, tries=7)}
        for i, (doc, q) in enumerate([("A", 0), ("B", 0), ("A", 1), ("A", 2), ("B", 1)], 1):
            text, qs, keys = docs[doc]
            assert long.check_level(text, qs, keys)
            qid, qtext, fmt, (kind, ans) = qs[q]
            d = Path(sys.argv[2]) / f"s{i}-{doc}-{qid}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "task.json").write_text(json.dumps({
                "id": d.name, "puzzle": d.name, "family": "swap", "level": 5, "mode": "2k", "kind": "long",
                "prompt": head + text + "\n```\n\nQuestion: " + qtext + long.TAIL.format(fmt=fmt), "answer": ans,
                "answer_kind": kind, "request": long.MODE, "chars": len(text)}))
            print(d.name, ans)
    else:
        sys.exit(__doc__)
