"""A growing coding conversation: each turn adds ~30K tokens of PolyStrata's own source and asks about it, so the
context fills like a coding agent's.  Per turn: the prompt's tokens, how many the engine read (the rest came from its
prompt cache), the prompt time, decode tok/s and expert-cache hits, from the server's /metrics.
Run inside the container: cd /opt/strata && .venv/bin/python /data/multiturn.py [order] [turns]
(order: fwd or rev, the file order, so two runs see different code)."""
import json, sys, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
CHUNK_CHARS = 100_000          # code is about 3.3 characters a token: ~30K tokens a turn
STOP_AT = 225_000              # no turn starts past this many prompt tokens (the context is 262,144)


def call(path, body=None):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=7200))


def chunks(order):
    files = sorted(p for d in ("src", "serve", "include") for p in Path("/opt/strata", d).rglob("*")
                   if p.suffix in (".cpp", ".hpp", ".cu", ".cuh", ".h", ".py") and p.stat().st_size < 200_000)
    if order == "rev":
        files.reverse()
    buf, names = [], []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        buf.append(f"=== {p.relative_to('/opt/strata')} ===\n{text}\n")
        names.append(str(p.relative_to("/opt/strata")))
        if sum(map(len, buf)) >= CHUNK_CHARS:
            yield "".join(buf), names
            buf, names = [], []


def main():
    order = sys.argv[1] if len(sys.argv) > 1 else "fwd"
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    call("/v1/chat/completions", {"model": "strata", "max_tokens": 32, "temperature": 0,
                                  "messages": [{"role": "user", "content": "Say hello."}]})   # warm-up, not counted
    msgs = [{"role": "system", "content": "You are a senior C++/CUDA engineer reviewing a codebase with the user."}]
    print(f"order {order}\nturn  prompt  read  prompt_s  read_tok/s  out  decode_tok/s  hits  wall_s")
    for i, (code, names) in enumerate(chunks(order), 1):
        if i > turns:
            break
        msgs.append({"role": "user", "content": f"Here is more of the codebase:\n\n{code}\n\nIn {names[-1]}, what "
                     "does the main function or class do, and is there a bug or risk in it? Be brief."})
        t0 = time.time()
        r = call("/v1/chat/completions", {"model": "strata", "messages": msgs, "max_tokens": 400, "temperature": 0})
        wall = time.time() - t0
        msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""})
        m = (call("/metrics").get("requests") or [{}])[0]
        read, ms = m.get("prompt_read"), m.get("prompt_ms") or 0
        print(f"{i:4d} {m.get('prompt_total'):7} {read:6} {ms / 1000:8.1f} {read / (ms / 1000) if ms else 0:10.0f} "
              f"{m.get('output_tokens'):4} {m.get('decode_tok_s'):12} {m.get('hit_rate'):5} {wall:7.1f}", flush=True)
        if (m.get("prompt_total") or 0) > STOP_AT:
            break


if __name__ == "__main__":
    main()
