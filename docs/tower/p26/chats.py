"""p26/chats.py TEMP [ROUNDS]: six short chats (code, explanation, story, table, bash, history) to the server on
127.0.0.1:8080 at temperature TEMP, 400 tokens each, thinking off; prints a line a chat.  The engine log has the
speed and the drafts offered and accepted."""
import json, sys, time, urllib.request
P = ["Write a Python function that merges overlapping intervals, with a docstring and three tests.",
     "Explain how a hash map handles collisions, for a new programmer.",
     "Write a short story about a lighthouse keeper who finds a message in a bottle.",
     "Make a markdown table comparing TCP, UDP and QUIC on five properties.",
     "Write a bash script that backs up a folder to a dated tar.gz and keeps the last seven.",
     "Summarize the causes of the First World War in about 300 words."]
def main():
    T = float(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for r in range(R):
        for i, p in enumerate(P):
            body = {"model": "strata", "messages": [{"role": "user", "content": p}], "max_tokens": 400,
                    "temperature": T, "top_p": 0.95, "seed": 100 + r, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t0 = time.time()
            try:
                u = json.load(urllib.request.urlopen(req, timeout=600)).get("usage", {})
                print(f"T{T} r{r} chat{i} out={u.get('completion_tokens')} {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                print(f"T{T} r{r} chat{i} error {e}", flush=True)


if __name__ == "__main__":
    main()
