"""Decode speed with draft acceptance: short prompt x4, then after an 11K prompt x2.
cd /opt/strata/terminal && PYTHONPATH=/opt/strata/terminal .venv/bin/python /data/dprobe.py LABEL"""
import json, random, sys, urllib.request
sys.path.insert(0, "/opt/strata")
from polystrata import long_prompt

Q = "Write a Python function that parses an INI file into a dict of dicts, with comments and tests."


def ask(messages, n):
    req = {"model": "strata", "messages": messages, "max_tokens": n, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"}),
        timeout=7200))
    return r, r.get("timings") or {}


def line(tag, t):
    d, a = t.get("draft_n") or 0, t.get("draft_n_accepted") or 0
    print(f"{sys.argv[1]} {tag}: tokens={t.get('predicted_n')} tok_s={t.get('predicted_per_second')} "
          f"ms_per_tok={t.get('predicted_per_token_ms')} drafts={d} accepted={a} ({100 * a / max(d, 1):.0f}%)", flush=True)


for i in range(4):
    line(f"short {i}", ask([{"role": "user", "content": f"({i}) {Q}"}], 400)[1])
for i in range(2):
    m = [{"role": "user", "content": f"Session {random.randrange(10**9)}.\n\n{long_prompt(8000)}"}]
    r, t = ask(m, 16)
    m += [{"role": "assistant", "content": r["choices"][0]["message"].get("content") or "OK"}, {"role": "user", "content": Q}]
    line(f"11k {i}", ask(m, 300)[1])
