"""Streaming on both APIs, thinking on and off: python /data/p21/stream.py [base_url]. Exit 1 on a miss."""
import json, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
bad = []


def events(path, body):
    req = urllib.request.Request(URL + path, json.dumps({**body, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    for raw in urllib.request.urlopen(req, timeout=600):
        line = raw.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            yield json.loads(line[5:])


def check(name, ok, detail):
    print(f"{'ok ' if ok else 'BAD'} {name}: {detail}", flush=True)
    if not ok:
        bad.append(name)


Q = [{"role": "user", "content": "What is 17 times 3? Answer with the number."}]
for think in (False, True):
    text, reason, n = "", "", 0
    for e in events("/v1/chat/completions", {"model": "strata", "max_tokens": 400 if think else 32, "temperature": 0,
                                             "messages": Q, "chat_template_kwargs": {"enable_thinking": think}}):
        d = (e.get("choices") or [{}])[0].get("delta", {})
        text += d.get("content") or ""
        reason += d.get("reasoning_content") or ""
        n += 1
    check(f"openai stream think={think}", n > 2 and "51" in text and (bool(reason) or not think),
          f"{n} chunks, reasoning {len(reason)} chars, content {text.strip()[:60]!r}")

for think in (False, True):
    text, reason, kinds = "", "", set()
    body = {"model": "strata", "max_tokens": 600 if think else 32, "messages": Q}
    body["thinking"] = {"type": "enabled", "budget_tokens": 400} if think else {"type": "disabled"}
    for e in events("/v1/messages", body):
        kinds.add(e.get("type"))
        d = e.get("delta") or {}
        text += d.get("text") or ""
        reason += d.get("thinking") or ""
    check(f"anthropic stream think={think}",
          {"message_start", "content_block_delta", "message_stop"} <= kinds and "51" in text and (bool(reason) or not think),
          f"events {sorted(kinds)}, thinking {len(reason)} chars, text {text.strip()[:60]!r}")

print("ALL OK" if not bad else f"MISSES: {bad}")
sys.exit(1 if bad else 0)
