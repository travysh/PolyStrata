"""p26/why400.py TASK.json: send a task's prompt as run_bench.py does and print the status and body, an error's too."""
import json, sys, time, urllib.error, urllib.request
t = json.load(open(sys.argv[1]))
body = {"model": "strata", "messages": [{"role": "user", "content": t["prompt"]}], "max_tokens": 16384,
        "reasoning_budget_tokens": 12288, "seed": 0, "stream": False, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time()
try:
    r = urllib.request.urlopen(req, timeout=3600)
    print(r.status, r.read()[:600].decode())
except urllib.error.HTTPError as e:
    print(e.code, e.read()[:2000].decode())
print("wall", round(time.time() - t0, 1))
