"""Decode speed after a long prompt (the decode attention kernel runs from 4,096 positions):
cd /opt/strata/terminal && PYTHONPATH=/opt/strata/terminal .venv/bin/python /data/ldecode.py LABEL [size ...]
Per size: one cold read, then a follow-up on the cached prompt that writes 300 tokens."""
import json, random, sys, time, urllib.request
sys.path.insert(0, "/opt/strata")
from polystrata import long_prompt


def ask(messages, n):
    req = {"model": "strata", "messages": messages, "max_tokens": n, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"}),
        timeout=7200))
    return r, r.get("timings") or {}


for s in sys.argv[2:] or ["24000"]:
    msgs = [{"role": "user", "content": f"Session {random.randrange(10**9)}.\n\n{long_prompt(int(s))}"}]
    r, t = ask(msgs, 16)
    msgs += [{"role": "assistant", "content": r["choices"][0]["message"].get("content") or "OK"},
             {"role": "user", "content": "Write a Python function that parses an INI file into a dict of dicts, with tests."}]
    r, t = ask(msgs, 300)
    print(f"{sys.argv[1]} size={s} prompt={r['usage']['prompt_tokens']} tokens={t.get('predicted_n')} "
          f"decode_tok_s={t.get('predicted_per_second')}", flush=True)
