"""Pass 22 scores from graded runs:  python3 score.py runs/TAG-cybertiel.graded.jsonl runs/TAG-inkling.graded.jsonl

Per model: tasks passed per group (repo fixes, Python, C++), as the mean over samples; thinking and answer tokens;
time; replies cut off at the token limit; then one row per task with each sample's result (P pass, . fail, L fail
cut off, A blocks that did not apply)."""
import json, sys
from collections import defaultdict
from pathlib import Path

# The server code quotes the models' own control tokens: Cyber-Tiel's (</think>, <tool_call>, <|im_start|>) are in the
# prompts of r01-r03, r05, r12-r14; Inkling's (<|end_message|>, <|message_model|>) in r12 and r13.  A model that writes
# its own control token while thinking ends its thinking there, so these are also counted apart.
NEUTRAL = {"r04", "r07", "r08", "r10", "r11", "r16"}
GROUPS = (("repo fixes", lambda i: i[0] == "r"), ("repo fixes with no control-token text", lambda i: i in NEUTRAL),
          ("Python functions", lambda i: i[0] == "p"), ("C++ functions", lambda i: i[0] == "c"))


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def mark(r):
    g = r["grade"]
    if g["pass"]:
        return "P"
    if r.get("finish") == "length":
        return "L"
    if g.get("applied") is False:
        return "A"
    return "."


def main():
    runs = {p: load(p) for p in sys.argv[1:]}
    names = {p: (rows[0]["model"] if rows else p) for p, rows in runs.items()}
    print("| | " + " | ".join(names.values()) + " |")
    print("|---|" + "---|" * len(runs))
    for label, pick in GROUPS:
        cells = []
        for rows in runs.values():
            mine = [r for r in rows if pick(r["id"])]
            ids = {r["id"] for r in mine}
            samples = {r["sample"] for r in mine} or {0}
            passed = sum(r["grade"]["pass"] for r in mine)
            cells.append(f"{passed / len(samples):.1f} of {len(ids)}" if ids else "-")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for label, fn in (("all tasks", lambda rows: f"{sum(r['grade']['pass'] for r in rows) / max(1, len({r['sample'] for r in rows})):.1f} of {len({r['id'] for r in rows})}"),
                      ("output tokens, median", lambda rows: str(sorted(r.get("completion_tokens") or 0 for r in rows)[len(rows) // 2])),
                      ("time per answer, median s", lambda rows: str(sorted(r.get("wall_s") or 0 for r in rows)[len(rows) // 2])),
                      ("time for all, h", lambda rows: f"{sum(r.get('wall_s') or 0 for r in rows) / 3600:.2f}"),
                      ("cut off at the limit", lambda rows: str(sum(r.get("finish") == "length" for r in rows))),
                      ("repo blocks that did not apply", lambda rows: str(sum(r["grade"].get("applied") is False for r in rows)))):
        print(f"| {label} | " + " | ".join(fn(rows) if rows else "-" for rows in runs.values()) + " |")
    print()
    per = defaultdict(dict)
    for p, rows in runs.items():
        for r in sorted(rows, key=lambda r: r["sample"]):
            per[r["id"]].setdefault(p, "")
            per[r["id"]][p] += mark(r)
    print("| task | " + " | ".join(names.values()) + " |")
    print("|---|" + "---|" * len(runs))
    for id in sorted(per):
        print(f"| {id} | " + " | ".join(per[id].get(p, "") for p in runs) + " |")


if __name__ == "__main__":
    main()
